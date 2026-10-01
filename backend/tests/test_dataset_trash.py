from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datasetui.api import create_router
import datasetui.dataset_trash as dataset_trash_module
from datasetui.config import Settings
from datasetui.database import Database, DatasetTrashConflictError
from datasetui.datasets import inspect_dataset, scan_storage_area
from datasetui.queueing import RecordingDispatcher


def _environment(tmp_path: Path):
    nas = tmp_path / "nas"
    for name in ("raw", "derived"):
        (nas / name).mkdir(parents=True)
    for name in ("cache", "staging", "jobs"):
        (tmp_path / name).mkdir()
    settings = Settings(
        database_path=tmp_path / "registry.sqlite3",
        redis_url="redis://unused",
        allowed_origins=("https://example.test",),
        nas_root=nas,
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        jobs_root=tmp_path / "jobs",
    )
    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Trash operator")
    dataset_root = nas / "raw" / "team" / "sample"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data").mkdir()
    (dataset_root / "data" / "part.bin").write_bytes(b"dataset-payload")
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "rby1",
                "total_episodes": 2,
                "total_frames": 20,
                "total_tasks": 1,
                "fps": 10,
                "features": {"action": {"dtype": "float32", "shape": [1]}},
            }
        )
    )
    record = inspect_dataset(
        area_root=nas / "raw", storage_area="raw", relative_path="team/sample"
    ).as_record()
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[record], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    app = FastAPI()
    app.include_router(
        create_router(database, RecordingDispatcher(), settings=settings)
    )
    return settings, database, profile, dataset, TestClient(app)


def _trash(client: TestClient, profile: dict, dataset: dict):
    return client.post(
        f"/api/v1/datasets/{dataset['id']}/trash",
        json={
            "profile_id": profile["id"],
            "expected_name": dataset["name"],
            "expected_fingerprint": dataset["fingerprint"],
        },
    )


def _restore(client: TestClient, profile: dict, dataset: dict):
    return client.post(
        f"/api/v1/dataset-trash/{dataset['id']}/restore",
        json={
            "profile_id": profile["id"],
            "expected_fingerprint": dataset["fingerprint"],
        },
    )


def test_trash_restore_preserves_identity_flags_and_defeats_stale_scan(tmp_path: Path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    flags = database.update_episode_flags(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        expected_revision=0,
        changes=[{"episode_index": 1, "flagged": True}],
    )
    annotations = database.replace_episode_annotations(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        episode_index=0,
        expected_revision=0,
        task_override="preserved task",
        atoms=[],
    )
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="preserved recipe",
        selection_mode="all",
    )
    snapshot = database.snapshot_curation_recipe(
        recipe["id"], profile_id=profile["id"]
    )
    validation, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"dataset_id": dataset["id"]},
        idempotency_key="completed-validation",
    )
    database.record_validation_run(
        job_id=validation["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="quick",
    )
    database.claim_job(validation["id"], worker_id="fixture-worker")
    database.fail_job(
        validation["id"], "fixture", "fixture", worker_id="fixture-worker"
    )

    response = _trash(client, profile, dataset)

    assert response.status_code == 200
    assert response.json()["state"] == "trashed"
    assert database.list_datasets() == []
    assert not (settings.nas_root / "raw" / "team" / "sample").exists()
    trashed = settings.nas_root / "raw" / ".datasetui-trash" / dataset["id"]
    assert trashed.is_dir()
    assert client.get(f"/api/v1/datasets/{dataset['id']}").status_code == 404
    trash_list = client.get(
        "/api/v1/dataset-trash", params={"profile_id": profile["id"]}
    )
    assert [item["dataset"]["id"] for item in trash_list.json()] == [dataset["id"]]

    stale_generation = database.begin_dataset_scan("raw")
    assert scan_storage_area(settings.nas_root, "raw", max_depth=6) == []
    restored = _restore(client, profile, dataset)
    assert restored.status_code == 200
    assert restored.json()["id"] == dataset["id"]
    stale = database.synchronize_datasets(
        storage_area="raw", records=[], scan_generation=stale_generation
    )
    assert stale["stale"] == 1
    assert database.list_datasets()[0]["id"] == dataset["id"]
    assert database.get_episode_flags(
        dataset_id=dataset["id"], profile_id=profile["id"]
    ) == flags
    assert database.get_episode_annotations(
        dataset_id=dataset["id"], profile_id=profile["id"], episode_index=0
    ) == annotations
    assert database.list_curation_recipes(
        dataset_id=dataset["id"], profile_id=profile["id"]
    )[0]["id"] == recipe["id"]
    assert database.get_curation_snapshot(snapshot["id"])["dataset_id"] == dataset["id"]
    assert database.list_validation_runs(dataset["id"])[0]["job_id"] == validation["id"]


def test_trash_requires_exact_confirmation_and_rejects_active_job(tmp_path: Path):
    _settings, database, profile, dataset, client = _environment(tmp_path)

    mismatch = client.post(
        f"/api/v1/datasets/{dataset['id']}/trash",
        json={
            "profile_id": profile["id"],
            "expected_name": dataset["name"] + "-wrong",
            "expected_fingerprint": dataset["fingerprint"],
        },
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["code"] == "confirmation_mismatch"

    job, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"dataset_id": dataset["id"]},
        idempotency_key="active-validation",
    )
    blocked = _trash(client, profile, dataset)
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "dataset_in_use"
    database.cancel_queued_job(job["id"], profile_id=profile["id"])
    assert _trash(client, profile, dataset).status_code == 200


def test_restore_never_overwrites_occupied_path_or_tampered_trash(tmp_path: Path):
    settings, _database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    occupied = settings.nas_root / "raw" / "team" / "sample"
    occupied.mkdir(parents=True)
    (occupied / "keep.txt").write_text("do-not-overwrite")

    blocked = _restore(client, profile, dataset)

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "restore_path_occupied"
    assert (occupied / "keep.txt").read_text() == "do-not-overwrite"
    assert (settings.nas_root / "raw" / ".datasetui-trash" / dataset["id"]).exists()
    (occupied / "keep.txt").unlink()
    occupied.rmdir()
    assert _restore(client, profile, dataset).status_code == 200


def test_restore_preserves_modified_contents_in_same_moved_directory(tmp_path: Path):
    settings, _database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    trashed = settings.nas_root / "raw" / ".datasetui-trash" / dataset["id"]
    (trashed / "data" / "part.bin").write_bytes(b"tampered")

    restored = _restore(client, profile, dataset)

    assert restored.status_code == 200
    assert not trashed.exists()
    assert (settings.nas_root / "raw" / "team" / "sample" / "data" / "part.bin").read_bytes() == b"tampered"


@pytest.mark.parametrize("occupied_kind", ["file", "symlink"])
def test_restore_leaf_conflicts_remain_recoverable(
    tmp_path: Path, occupied_kind: str
):
    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    target = settings.nas_root / "raw" / "team" / "sample"
    if occupied_kind == "file":
        target.write_text("foreign-file")
        foreign = target
    else:
        foreign = tmp_path / "foreign-target"
        foreign.write_text("foreign-symlink-target")
        target.symlink_to(foreign)

    blocked = _restore(client, profile, dataset)
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "restore_path_occupied"
    assert database.get_dataset_trash(dataset["id"])["state"] == "trashed"
    assert foreign.read_text().startswith("foreign-")
    target.unlink()
    assert _restore(client, profile, dataset).status_code == 200
    if occupied_kind == "symlink":
        assert foreign.read_text() == "foreign-symlink-target"


def test_restore_rejects_substituted_trash_root_identity(tmp_path: Path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    trashed = settings.nas_root / "raw" / ".datasetui-trash" / dataset["id"]
    retained = trashed.with_name(f"{dataset['id']}-retained")
    trashed.rename(retained)
    trashed.mkdir()
    (trashed / "foreign.txt").write_text("foreign")

    blocked = _restore(client, profile, dataset)

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "trash_recovery_required"
    assert database.get_dataset_trash(dataset["id"])["state"] == "recovery_required"
    assert (retained / "data" / "part.bin").read_bytes() == b"dataset-payload"
    assert (trashed / "foreign.txt").read_text() == "foreign"


def test_symlink_ancestor_and_nested_registry_are_rejected(tmp_path: Path):
    settings, database, profile, _dataset, client = _environment(tmp_path)
    outside = tmp_path / "outside"
    (outside / "ds" / "meta").mkdir(parents=True)
    (outside / "ds" / "data").mkdir()
    (outside / "ds" / "meta" / "info.json").write_text("{}")
    (settings.nas_root / "raw" / "alias").symlink_to(outside)
    fake = {
        "storage_area": "raw",
        "relative_path": "alias/ds",
        "name": "aliased",
        "codebase_version": "v3.0",
        "readiness": "ready",
        "robot_type": "rby1",
        "total_episodes": 1,
        "total_frames": 1,
        "total_tasks": 1,
        "fps": 10,
        "fingerprint": "a" * 64,
        "info_mtime_ns": 1,
        "info_size": 2,
        "scan_error": None,
    }
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[fake], scan_generation=generation
    )
    aliased = next(item for item in database.list_datasets() if item["name"] == "aliased")

    blocked = _trash(client, profile, aliased)

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "trash_path_conflict"
    assert (outside / "ds").exists()


def test_moving_marker_blocks_job_creation_and_scan(tmp_path: Path):
    _settings, database, profile, dataset, _client = _environment(tmp_path)
    database.prepare_dataset_trash(
        dataset["id"],
        profile_id=profile["id"],
        expected_name=dataset["name"],
        expected_fingerprint=dataset["fingerprint"],
    )

    with pytest.raises(DatasetTrashConflictError):
        database.create_job(
            kind="phase2.smoke",
            queue_name="cpu",
            profile_id=profile["id"],
            payload={},
            idempotency_key="blocked-during-move",
        )
    with pytest.raises(DatasetTrashConflictError):
        database.begin_dataset_scan("raw")


def test_operation_owner_and_in_progress_retry_contract(tmp_path: Path):
    _settings, database, profile, dataset, client = _environment(tmp_path)
    database.prepare_dataset_trash(
        dataset["id"],
        profile_id=profile["id"],
        expected_name=dataset["name"],
        expected_fingerprint=dataset["fingerprint"],
    )
    retry = _trash(client, profile, dataset)
    assert retry.status_code == 409
    assert retry.json()["detail"]["code"] == "trash_operation_in_progress"

    database.abort_dataset_trash(dataset["id"])
    assert _trash(client, profile, dataset).status_code == 200
    restorer = database.create_profile("Restorer")
    record = database.prepare_dataset_restore(
        dataset["id"],
        profile_id=restorer["id"],
        expected_fingerprint=dataset["fingerprint"],
    )
    assert record["requested_by_profile_id"] == restorer["id"]


def test_atomic_trash_move_does_not_replace_racing_destination(
    tmp_path: Path, monkeypatch
):
    settings, _database, profile, dataset, client = _environment(tmp_path)
    def create_competing_destination(_source, destination):
        destination.mkdir()
        (destination / "owner.txt").write_text("competing-owner")

    monkeypatch.setattr(
        dataset_trash_module, "_before_rename_hook", create_competing_destination
    )

    blocked = _trash(client, profile, dataset)

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "trash_path_conflict"
    assert (settings.nas_root / "raw" / "team" / "sample").is_dir()
    competing = settings.nas_root / "raw" / ".datasetui-trash" / dataset["id"]
    assert (competing / "owner.txt").read_text() == "competing-owner"


def test_atomic_restore_does_not_replace_racing_destination(
    tmp_path: Path, monkeypatch
):
    settings, _database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    def create_competing_destination(_source, destination):
        destination.mkdir(parents=True)
        (destination / "owner.txt").write_text("competing-owner")

    monkeypatch.setattr(
        dataset_trash_module, "_before_rename_hook", create_competing_destination
    )

    blocked = _restore(client, profile, dataset)

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "restore_path_occupied"
    original = settings.nas_root / "raw" / "team" / "sample"
    assert (original / "owner.txt").read_text() == "competing-owner"
    assert (settings.nas_root / "raw" / ".datasetui-trash" / dataset["id"]).is_dir()


def test_trash_rejects_parent_swap_to_symlink_during_rename(tmp_path: Path, monkeypatch):
    settings, database, profile, dataset, client = _environment(tmp_path)
    outside = tmp_path / "outside-race"
    outside.mkdir()
    original_parent = settings.nas_root / "raw" / "team"
    retained_parent = settings.nas_root / "raw" / "team-retained"

    def swap_parent(_source, _destination):
        original_parent.rename(retained_parent)
        original_parent.symlink_to(outside)

    monkeypatch.setattr(dataset_trash_module, "_before_rename_hook", swap_parent)
    blocked = _trash(client, profile, dataset)

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] in {
        "trash_path_conflict",
        "trash_recovery_required",
    }
    assert database.get_dataset_trash(dataset["id"])["state"] == "recovery_required"
    assert (retained_parent / "sample" / "data" / "part.bin").read_bytes() == b"dataset-payload"
    assert list(outside.iterdir()) == []


def test_stale_marker_and_post_rename_retries_clear_global_barrier(tmp_path: Path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    database.prepare_dataset_trash(
        dataset["id"],
        profile_id=profile["id"],
        expected_name=dataset["name"],
        expected_fingerprint=dataset["fingerprint"],
    )
    source_device, source_inode = dataset_trash_module.registered_dataset_identity(
        settings.nas_root,
        storage_area=dataset["storage_area"],
        relative_path=dataset["relative_path"],
    )
    database.record_dataset_trash_source_identity(
        dataset["id"], source_device=source_device, source_inode=source_inode
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE dataset_trash SET updated_at = '2000-01-01T00:00:00.000Z' WHERE dataset_id = ?",
            (dataset["id"],),
        )

    assert _trash(client, profile, dataset).status_code == 200
    assert database.begin_dataset_scan("raw") >= 1

    restore_record = database.prepare_dataset_restore(
        dataset["id"],
        profile_id=profile["id"],
        expected_fingerprint=dataset["fingerprint"],
    )
    dataset_trash_module.restore_dataset_from_trash(settings.nas_root, restore_record)
    with database.connect() as connection:
        connection.execute(
            "UPDATE dataset_trash SET updated_at = '2000-01-01T00:00:00.000Z' WHERE dataset_id = ?",
            (dataset["id"],),
        )

    restored = _restore(client, profile, dataset)
    assert restored.status_code == 200
    assert restored.json()["id"] == dataset["id"]
    job, created = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"dataset_id": dataset["id"]},
        idempotency_key="barrier-cleared",
    )
    assert created and job["status"] == "queued"


def test_hf_import_cannot_republish_reserved_trashed_revision(tmp_path: Path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    commit_sha = "a" * 40
    target = (
        settings.nas_root
        / "raw"
        / "hf"
        / "rainbowrobotics"
        / "sample"
        / "revisions"
        / commit_sha
    )
    target.parent.mkdir(parents=True)
    (settings.nas_root / "raw" / "team" / "sample").rename(target)
    relative_path = target.relative_to(settings.nas_root / "raw").as_posix()
    with database.connect() as connection:
        connection.execute(
            "UPDATE datasets SET relative_path = ? WHERE id = ?",
            (relative_path, dataset["id"]),
        )
    dataset = database.get_dataset(dataset["id"])
    assert _trash(client, profile, dataset).status_code == 200

    with pytest.raises(DatasetTrashConflictError) as blocked:
        database.create_hf_import_job(
            profile_id=profile["id"],
            repo_id="rainbowrobotics/sample",
            dataset_name="sample",
            requested_revision="main",
            commit_sha=commit_sha,
            expected_file_count=1,
            expected_total_bytes=1,
            idempotency_key="same-trashed-revision",
        )
    assert blocked.value.code == "dataset_in_use"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM hf_sources").fetchone()[0] == 0

    job, created = database.create_hf_import_job(
        profile_id=profile["id"],
        repo_id="rainbowrobotics/sample",
        dataset_name="sample",
        requested_revision="other",
        commit_sha="b" * 40,
        expected_file_count=1,
        expected_total_bytes=1,
        idempotency_key="different-revision",
    )
    assert created and job["status"] == "queued"
