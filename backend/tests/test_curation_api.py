from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from datasetui.database import Database


def _record(*, fingerprint: str = "source-revision-a") -> dict[str, Any]:
    return {
        "storage_area": "raw",
        "relative_path": "lab/pick-cup",
        "name": "pick-cup",
        "codebase_version": "v3.0",
        "readiness": "ready",
        "robot_type": "rby1",
        "total_episodes": 5,
        "total_frames": 150,
        "total_tasks": 1,
        "fps": 30,
        "fingerprint": fingerprint,
        "info_mtime_ns": 1,
        "info_size": 100,
        "scan_error": None,
    }


def _register(database: Database, *, fingerprint: str = "source-revision-a") -> dict:
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[_record(fingerprint=fingerprint)],
        scan_generation=generation,
    )
    return database.list_datasets()[0]


def _profile(client: TestClient, name: str) -> dict:
    response = client.post("/api/v1/profiles", json={"name": name})
    assert response.status_code == 201
    return response.json()


def test_relative_recipe_dispatch_preserves_mask_chunk_and_large_timeout(
    client, database, monkeypatch
):
    from datasetui.queueing import RecordingDispatcher

    seen = []
    original = RecordingDispatcher.enqueue

    def capture(self, **kwargs):
        seen.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(RecordingDispatcher, "enqueue", capture)
    dataset = _register(database)
    profile = _profile(client, "Relative timeout")
    response = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Selected relative",
            "selection_mode": "all",
            "relative_action": {
                "enabled": True,
                "dimensions": ["joint_1"],
                "chunk_size": 40,
            },
        },
    )
    assert response.status_code == 201
    recipe = response.json()
    started = client.post(
        f"/api/v1/recipes/{recipe['id']}/runs",
        json={
            "profile_id": profile["id"],
            "output_name": "relative-test",
            "idempotency_key": "relative-timeout",
        },
    )
    assert started.status_code == 202
    snapshot = database.get_curation_snapshot(started.json()["payload"]["snapshot_id"])
    assert snapshot["relative_action"] == {
        "enabled": True,
        "dimensions": ["joint_1"],
        "chunk_size": 40,
    }
    assert seen[-1]["job_timeout"] == 86400


def test_flags_are_profile_scoped_revision_safe_and_strict(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    kim = _profile(client, "Kim")
    lee = _profile(client, "Lee")
    url = f"/api/v1/datasets/{dataset['id']}/flags"

    empty = client.get(url, params={"profile_id": kim["id"]})
    assert empty.status_code == 200
    assert empty.json()["revision"] == 0
    assert empty.json()["episode_indices"] == []

    updated = client.patch(
        url,
        json={
            "profile_id": kim["id"],
            "expected_revision": 0,
            "changes": [
                {"episode_index": 3, "flagged": True},
                {"episode_index": 1, "flagged": True},
            ],
        },
    )
    assert updated.status_code == 200
    assert updated.json()["revision"] == 1
    assert updated.json()["episode_indices"] == [1, 3]
    assert (
        client.get(url, params={"profile_id": lee["id"]}).json()["episode_indices"]
        == []
    )

    stale = client.patch(
        url,
        json={
            "profile_id": kim["id"],
            "expected_revision": 0,
            "changes": [{"episode_index": 2, "flagged": True}],
        },
    )
    assert stale.status_code == 409
    assert "revision" not in stale.text.lower()

    outside = client.patch(
        url,
        json={
            "profile_id": kim["id"],
            "expected_revision": 1,
            "changes": [{"episode_index": 5, "flagged": True}],
        },
    )
    assert outside.status_code == 422
    secret = client.patch(
        url,
        json={
            "profile_id": kim["id"],
            "expected_revision": 1,
            "changes": [{"episode_index": 2, "flagged": True}],
            "password": "must-not-be-accepted",
        },
    )
    assert secret.status_code == 422


def test_annotations_are_profile_scoped_revision_safe_and_snapshot_frozen(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    kim = _profile(client, "Annotation Kim")
    lee = _profile(client, "Annotation Lee")
    url = f"/api/v1/datasets/{dataset['id']}/annotations/2"
    atoms = [
        {
            "role": "assistant",
            "content": "grasp the cup",
            "style": "subtask",
            "timestamp": 0.3,
            "camera": None,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": None,
            "style": None,
            "timestamp": 0.5,
            "camera": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "say", "arguments": {"text": "done"}},
                }
            ],
        },
        {
            "role": "assistant",
            "content": '{"label":"cup","count":1}',
            "style": "vqa",
            "timestamp": 0.7,
            "camera": "observation.images.top",
            "tool_calls": None,
        },
    ]

    empty = client.get(url, params={"profile_id": kim["id"]})
    assert empty.status_code == 200
    assert empty.json()["revision"] == 0
    saved = client.put(
        url,
        json={
            "profile_id": kim["id"],
            "expected_revision": 0,
            "task_override": "place the cup",
            "atoms": atoms,
        },
    )
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    assert saved.json()["task_override"] == "place the cup"
    assert client.get(url, params={"profile_id": lee["id"]}).json()["atoms"] == []

    stale = client.put(
        url,
        json={
            "profile_id": kim["id"],
            "expected_revision": 0,
            "task_override": None,
            "atoms": [],
        },
    )
    assert stale.status_code == 409

    recipe = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": kim["id"],
            "name": "Annotated subset",
            "selection_mode": "all",
            "include_annotations": True,
        },
    ).json()
    snapshot = client.post(
        f"/api/v1/recipes/{recipe['id']}/snapshots",
        json={"profile_id": kim["id"]},
    ).json()
    assert snapshot["include_annotations"] is True
    assert snapshot["annotation_episode_indices"] == [2]

    changed = client.put(
        url,
        json={
            "profile_id": kim["id"],
            "expected_revision": 1,
            "task_override": "changed later",
            "atoms": [],
        },
    )
    assert changed.status_code == 200
    frozen = database.get_curation_snapshot_annotations(snapshot["id"])[2]
    assert frozen["revision"] == 1
    assert frozen["task_override"] == "place the cup"
    assert frozen["atoms"] == atoms


def test_annotations_reject_invalid_or_unsafe_payloads(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Strict annotator")
    url = f"/api/v1/datasets/{dataset['id']}/annotations/0"
    base = {
        "profile_id": profile["id"],
        "expected_revision": 0,
        "task_override": None,
    }
    invalid_atoms = [
        {
            "role": "assistant",
            "content": "not-json",
            "style": "vqa",
            "timestamp": 0,
            "camera": "observation.images.top",
            "tool_calls": None,
        },
        {
            "role": "user",
            "content": "camera leak",
            "style": "subtask",
            "timestamp": 0,
            "camera": "observation.images.top",
            "tool_calls": None,
        },
    ]
    for atom in invalid_atoms:
        assert client.put(url, json={**base, "atoms": [atom]}).status_code == 422
    assert (
        client.put(url, json={**base, "atoms": [], "password": "no"}).status_code == 422
    )
    assert (
        client.get(
            f"/api/v1/datasets/{dataset['id']}/annotations/5",
            params={"profile_id": profile["id"]},
        ).status_code
        == 422
    )


def test_annotations_do_not_cross_dataset_revisions(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Revision annotator")
    url = f"/api/v1/datasets/{dataset['id']}/annotations/1"
    assert (
        client.put(
            url,
            json={
                "profile_id": profile["id"],
                "expected_revision": 0,
                "task_override": "old revision",
                "atoms": [],
            },
        ).status_code
        == 200
    )
    _register(database, fingerprint="source-revision-b")
    current = client.get(url, params={"profile_id": profile["id"]}).json()
    assert current["revision"] == 0
    assert current["task_override"] is None


def test_recipe_snapshot_freezes_selection_and_flags(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Researcher")
    flags_url = f"/api/v1/datasets/{dataset['id']}/flags"
    assert (
        client.patch(
            flags_url,
            json={
                "profile_id": profile["id"],
                "expected_revision": 0,
                "changes": [
                    {"episode_index": 1, "flagged": True},
                    {"episode_index": 4, "flagged": True},
                ],
            },
        ).status_code
        == 200
    )

    recipes_url = f"/api/v1/datasets/{dataset['id']}/recipes"
    created = client.post(
        recipes_url,
        json={
            "profile_id": profile["id"],
            "name": "  Review   failures  ",
            "selection_mode": "flagged",
        },
    )
    assert created.status_code == 201
    recipe = created.json()
    assert recipe["name"] == "Review failures"

    duplicate = client.post(
        recipes_url,
        json={
            "profile_id": profile["id"],
            "name": "review failures",
            "selection_mode": "all",
        },
    )
    assert duplicate.status_code == 409

    first = client.post(
        f"/api/v1/recipes/{recipe['id']}/snapshots",
        json={"profile_id": profile["id"]},
    )
    assert first.status_code == 201
    assert first.json()["flag_revision"] == 1
    assert first.json()["flagged_episode_indices"] == [1, 4]
    assert first.json()["selected_episode_indices"] == [1, 4]

    changed = client.patch(
        flags_url,
        json={
            "profile_id": profile["id"],
            "expected_revision": 1,
            "changes": [
                {"episode_index": 1, "flagged": False},
                {"episode_index": 3, "flagged": True},
            ],
        },
    )
    assert changed.status_code == 200
    second = client.post(
        f"/api/v1/recipes/{recipe['id']}/snapshots",
        json={"profile_id": profile["id"]},
    )
    assert second.json()["flag_revision"] == 2
    assert second.json()["flagged_episode_indices"] == [3, 4]
    assert first.json()["selected_episode_indices"] == [1, 4]


def test_random_train_eval_split_is_reproducible_and_snapshot_frozen(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Random splitter")
    recipe_response = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Seeded split",
            "selection_mode": "all",
            "operation": "train_eval_split",
            "split_config": {
                "method": "random",
                "eval_percent": 40,
                "seed": 2026,
            },
        },
    )
    assert recipe_response.status_code == 201
    recipe = recipe_response.json()
    assert recipe["split_config"] == {
        "method": "random",
        "eval_percent": 40.0,
        "seed": 2026,
    }

    snapshots = [
        client.post(
            f"/api/v1/recipes/{recipe['id']}/snapshots",
            json={"profile_id": profile["id"]},
        ).json()
        for _ in range(2)
    ]
    assert snapshots[0]["eval_episode_indices"] == snapshots[1]["eval_episode_indices"]
    assert len(snapshots[0]["eval_episode_indices"]) == 2
    assert set(snapshots[0]["eval_episode_indices"]).issubset(
        snapshots[0]["selected_episode_indices"]
    )
    frozen = database.get_curation_snapshot(snapshots[0]["id"])
    assert frozen["eval_episode_indices"] == snapshots[0]["eval_episode_indices"]


def test_random_train_eval_split_supports_zero_and_one_hundred_percent(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Ratio edges")
    for percent, expected in ((0, []), (100, [0, 1, 2, 3, 4])):
        recipe = client.post(
            f"/api/v1/datasets/{dataset['id']}/recipes",
            json={
                "profile_id": profile["id"],
                "name": f"Eval {percent}",
                "selection_mode": "all",
                "operation": "train_eval_split",
                "split_config": {
                    "method": "random",
                    "eval_percent": percent,
                    "seed": 7,
                },
            },
        ).json()
        snapshot = client.post(
            f"/api/v1/recipes/{recipe['id']}/snapshots",
            json={"profile_id": profile["id"]},
        ).json()
        assert snapshot["eval_episode_indices"] == expected

    for percent, expected_count in ((9, 0), (10, 1)):
        recipe = client.post(
            f"/api/v1/datasets/{dataset['id']}/recipes",
            json={
                "profile_id": profile["id"],
                "name": f"Rounded Eval {percent}",
                "selection_mode": "all",
                "operation": "train_eval_split",
                "split_config": {
                    "method": "random",
                    "eval_percent": percent,
                    "seed": 11,
                },
            },
        ).json()
        snapshot = client.post(
            f"/api/v1/recipes/{recipe['id']}/snapshots",
            json={"profile_id": profile["id"]},
        ).json()
        assert len(snapshot["eval_episode_indices"]) == expected_count

    invalid = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Invalid ratio",
            "selection_mode": "all",
            "operation": "train_eval_split",
            "split_config": {
                "method": "random",
                "eval_percent": 100.1,
                "seed": 0,
            },
        },
    )
    assert invalid.status_code == 422


def test_random_train_eval_split_partitions_the_selected_episode_scope(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Scoped splitter")
    client.patch(
        f"/api/v1/datasets/{dataset['id']}/flags",
        json={
            "profile_id": profile["id"],
            "expected_revision": 0,
            "changes": [
                {"episode_index": 1, "flagged": True},
                {"episode_index": 4, "flagged": True},
            ],
        },
    )
    recipe = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Flag scope random split",
            "selection_mode": "flagged",
            "operation": "train_eval_split",
            "split_config": {
                "method": "random",
                "eval_percent": 50,
                "seed": 3,
            },
        },
    ).json()
    snapshot = client.post(
        f"/api/v1/recipes/{recipe['id']}/snapshots",
        json={"profile_id": profile["id"]},
    ).json()
    assert snapshot["selected_episode_indices"] == [1, 4]
    assert len(snapshot["eval_episode_indices"]) == 1
    assert set(snapshot["eval_episode_indices"]).issubset({1, 4})


def test_recipes_and_flags_do_not_cross_dataset_revisions(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Researcher")
    flags_url = f"/api/v1/datasets/{dataset['id']}/flags"
    client.patch(
        flags_url,
        json={
            "profile_id": profile["id"],
            "expected_revision": 0,
            "changes": [{"episode_index": 2, "flagged": True}],
        },
    )
    recipe = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Keep all",
            "selection_mode": "all",
        },
    ).json()

    _register(database, fingerprint="source-revision-b")
    current_flags = client.get(flags_url, params={"profile_id": profile["id"]})
    assert current_flags.json()["episode_indices"] == []
    assert current_flags.json()["dataset_fingerprint"] == "source-revision-b"
    assert (
        client.get(
            f"/api/v1/datasets/{dataset['id']}/recipes",
            params={"profile_id": profile["id"]},
        ).json()
        == []
    )
    stale_snapshot = client.post(
        f"/api/v1/recipes/{recipe['id']}/snapshots",
        json={"profile_id": profile["id"]},
    )
    assert stale_snapshot.status_code == 409


def test_recipe_update_is_owner_scoped_and_archivable(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    owner = _profile(client, "Owner")
    other = _profile(client, "Other")
    recipe = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": owner["id"],
            "name": "Unflagged only",
            "selection_mode": "unflagged",
        },
    ).json()

    assert (
        client.patch(
            f"/api/v1/recipes/{recipe['id']}",
            json={"profile_id": other["id"], "archived": True},
        ).status_code
        == 404
    )
    archived = client.patch(
        f"/api/v1/recipes/{recipe['id']}",
        json={"profile_id": owner["id"], "archived": True},
    )
    assert archived.status_code == 200
    assert archived.json()["archived_at"] is not None
    listed = client.get(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        params={"profile_id": owner["id"]},
    )
    assert listed.json() == []


def test_recipe_run_freezes_transform_config_and_is_idempotent(
    client: TestClient, database: Database
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Runner")
    client.patch(
        f"/api/v1/datasets/{dataset['id']}/flags",
        json={
            "profile_id": profile["id"],
            "expected_revision": 0,
            "changes": [{"episode_index": 4, "flagged": True}],
        },
    )
    recipe = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Split with trim",
            "selection_mode": "all",
            "operation": "train_eval_split",
            "trim_config": {
                "enabled": True,
                "threshold": 0.03,
                "hold_time_s": 0.5,
                "margin_s": 1.0,
                "start_hold_time_s": 0.2,
                "end_hold_time_s": 0.7,
                "start_margin_s": 0,
                "end_margin_s": 2.0,
                "dimensions": ["joint_0"],
                "episode_overrides": {"4": {"start_frame": 2, "end_frame": 20}},
            },
        },
    ).json()

    body = {
        "profile_id": profile["id"],
        "output_name": "pick-cup-clean",
        "idempotency_key": "phase8-run-1",
    }
    created = client.post(f"/api/v1/recipes/{recipe['id']}/runs", json=body)
    assert created.status_code == 202
    assert created.json()["kind"] == "curation.materialize"
    assert set(created.json()["payload"]) == {"snapshot_id", "output_name"}
    snapshot = database.get_curation_snapshot(created.json()["payload"]["snapshot_id"])
    assert snapshot["operation"] == "train_eval_split"
    assert snapshot["flagged_episode_indices"] == [4]
    assert snapshot["trim_config"]["episode_overrides"]["4"]["start_frame"] == 2
    assert snapshot["trim_config"]["start_hold_time_s"] == 0.2
    assert snapshot["trim_config"]["end_hold_time_s"] == 0.7
    assert snapshot["trim_config"]["start_margin_s"] == 0
    assert snapshot["trim_config"]["end_margin_s"] == 2.0

    repeated = client.post(f"/api/v1/recipes/{recipe['id']}/runs", json=body)
    assert repeated.status_code == 200
    assert repeated.json()["id"] == created.json()["id"]
    rejected = client.post(
        f"/api/v1/recipes/{recipe['id']}/runs",
        json={**body, "output_name": "different"},
    )
    assert rejected.status_code == 409

    secret = client.post(
        f"/api/v1/recipes/{recipe['id']}/runs",
        json={**body, "idempotency_key": "secret", "password": "nope"},
    )
    assert secret.status_code == 422


def test_concurrent_idempotent_recipe_run_reconciles_the_winning_snapshot(
    client: TestClient, database: Database, monkeypatch: Any
) -> None:
    dataset = _register(database)
    profile = _profile(client, "Concurrent runner")
    recipe = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Concurrent subset",
            "selection_mode": "all",
        },
    ).json()
    original_create_job = database.create_job

    def race_create_job(**kwargs: Any) -> tuple[dict[str, Any], bool]:
        winner_snapshot = database.snapshot_curation_recipe(
            recipe["id"], profile_id=profile["id"]
        )
        original_create_job(
            **{
                **kwargs,
                "payload": {
                    "snapshot_id": winner_snapshot["id"],
                    "output_name": kwargs["payload"]["output_name"],
                },
            }
        )
        return original_create_job(**kwargs)

    monkeypatch.setattr(database, "create_job", race_create_job)
    response = client.post(
        f"/api/v1/recipes/{recipe['id']}/runs",
        json={
            "profile_id": profile["id"],
            "output_name": "concurrent-output",
            "idempotency_key": "concurrent-run-key",
        },
    )

    assert response.status_code == 200
    job = response.json()
    assert job["kind"] == "curation.materialize"
    assert job["payload"]["output_name"] == "concurrent-output"
    assert (
        database.get_curation_snapshot(job["payload"]["snapshot_id"])["recipe_id"]
        == recipe["id"]
    )
