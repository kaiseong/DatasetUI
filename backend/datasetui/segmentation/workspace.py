"""Per-video confirmed objects (editing state, never an approval)."""

from __future__ import annotations

import json
from uuid import UUID

from pydantic import Field, model_validator

from datasetui.database import utc_now
from datasetui.segmentation.catalog import dataset_catalog
from datasetui.segmentation.contract import (
    Correction,
    FrameIndex,
    RegionPrompt,
    StrictModel,
)


class WorkspaceScope(StrictModel):
    profile_id: UUID
    dataset_id: UUID
    episode_index: FrameIndex
    video_key: str = Field(min_length=1, max_length=240, pattern=r"^[A-Za-z0-9_.-]+$")


class ConfirmedObject(StrictModel):
    object_id: int = Field(ge=1, le=32)
    name: str = Field(min_length=1, max_length=120)
    target: str = Field(pattern=r"^(protect|replace)$")
    prompts: list[RegionPrompt] = Field(default_factory=list, max_length=32)
    corrections: list[Correction] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def valid_object(self):
        from types import SimpleNamespace

        from datasetui.segmentation.contract import (
            seed_brush_objects,
            validate_initial_guidance,
        )
        self.name = self.name.strip()
        if not self.name or not (self.prompts or self.corrections):
            raise ValueError("객체 이름과 지시가 필요합니다.")
        if any(item.object_id != self.object_id or item.target != self.target
               for item in [*self.prompts, *self.corrections]):
            raise ValueError("객체 번호와 남김/제거 설정이 일치하지 않습니다.")
        semantic = [p for p in self.prompts if p.text]
        if len({p.text for p in semantic}) > 1:
            raise ValueError("동일 객체 그룹에는 하나의 Instruction을 사용하세요.")
        refs = [r.candidate_id for p in semantic for r in p.selected_candidates]
        if len(refs) != len(set(refs)):
            raise ValueError("동일 후보가 중복 선택되었습니다.")
        # Text without chosen candidates is valid: candidates are then chosen
        # (or auto-selected when unambiguous) on the episode preview.
        if any(item.member_candidate_id and item.member_candidate_id not in refs for item in [*self.prompts, *self.corrections]):
            raise ValueError("선택하지 않은 후보의 보정이 남아 있습니다. 해당 보정을 지우세요.")
        if refs:
            geometry = [p for p in self.prompts if p.points or p.box is not None]
            for item in [*geometry, *self.corrections]:
                if item.member_candidate_id is None:
                    if len(refs) != 1:
                        raise ValueError("보정할 그룹 내 후보를 선택한 뒤 확인하세요.")
                    # Singleton binding is resolved at inference; keep the
                    # semantic prompt separate from member-specific UI hints.
        self.prompts.sort(key=lambda p: not bool(p.text))
        prompts = seed_brush_objects(self.prompts, self.corrections)
        validate_initial_guidance(SimpleNamespace(mode="object_selection", prompts=prompts))
        return self


class WorkspaceSave(WorkspaceScope):
    expected_revision: int = Field(ge=0, strict=True)
    metadata_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    objects: list[ConfirmedObject] = Field(max_length=32)

    @model_validator(mode="after")
    def unique_objects(self):
        if len({o.object_id for o in self.objects}) != len(self.objects):
            raise ValueError("객체 번호가 중복되었습니다.")
        if sum(len(o.prompts) for o in self.objects) > 32 or sum(len(o.corrections) for o in self.objects) > 100:
            raise ValueError("지시 개수가 처리 한도를 넘었습니다.")
        if len(self.model_dump_json().encode()) > 1024 * 1024:
            raise ValueError("객체 설정은 1 MiB 이하여야 합니다.")
        return self


def initialize(database):
    with database.connect() as connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS segmentation_workspaces (
            profile_id TEXT NOT NULL REFERENCES profiles(id),
            dataset_id TEXT NOT NULL REFERENCES datasets(id),
            episode_index INTEGER NOT NULL, video_key TEXT NOT NULL,
            revision INTEGER NOT NULL, metadata_revision TEXT NOT NULL,
            objects_json TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(profile_id,dataset_id,episode_index,video_key))''')


def _scope(database, settings, payload):
    database.get_profile(str(payload.profile_id))
    catalog = dataset_catalog(database, settings, str(payload.dataset_id))
    if payload.video_key not in catalog["video_keys"] or payload.episode_index >= catalog["total_episodes"]:
        raise ValueError("원본에 없는 에피소드·카메라입니다.")
    return catalog, (str(payload.profile_id), str(payload.dataset_id), payload.episode_index, payload.video_key)


def read_workspace(database, settings, payload):
    catalog, key = _scope(database, settings, payload)
    with database.connect() as connection:
        row = connection.execute('''SELECT * FROM segmentation_workspaces WHERE
            profile_id=? AND dataset_id=? AND episode_index=? AND video_key=?''', key).fetchone()
    return {"revision": row["revision"] if row else 0,
            "metadata_revision": catalog["metadata_revision"],
            "stale": bool(row and row["metadata_revision"] != catalog["metadata_revision"]),
            "objects": json.loads(row["objects_json"]) if row else []}


def save_workspace(database, settings, payload):
    catalog, key = _scope(database, settings, payload)
    if catalog["metadata_revision"] != payload.metadata_revision:
        raise ValueError("원본 메타정보가 변경되었습니다. 다시 불러오세요.")
    from types import SimpleNamespace

    from datasetui.segmentation.candidates import validate_candidate_references
    validate_candidate_references(database, settings, SimpleNamespace(
        dataset_id=payload.dataset_id, episode_index=payload.episode_index,
        video_key=payload.video_key, prompts=[p for o in payload.objects for p in o.prompts]), str(payload.profile_id))
    encoded = json.dumps([o.model_dump(mode="json") for o in payload.objects], ensure_ascii=False)
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute('''SELECT revision FROM segmentation_workspaces WHERE
            profile_id=? AND dataset_id=? AND episode_index=? AND video_key=?''', key).fetchone()
        if (row["revision"] if row else 0) != payload.expected_revision:
            raise ValueError("다른 화면에서 객체가 변경되었습니다. 초안을 보존한 채 다시 불러오세요.")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "segmentation_approvals" in tables:
            connection.execute("""DELETE FROM segmentation_approvals WHERE profile_id=? AND preview_id IN (
                SELECT id FROM jobs WHERE profile_id=? AND json_extract(payload_json,'$.spec.dataset_id')=?
                AND json_extract(payload_json,'$.spec.episode_index')=? AND json_extract(payload_json,'$.spec.video_key')=?)""",
                (key[0], *key))
        if "segmentation_batch_items" in tables:
            connection.execute("""UPDATE segmentation_batch_items SET approved_recipe_hash=NULL,
                approved_artifact_fingerprint=NULL, approved_token_hash=NULL, approved_at=NULL
                WHERE episode_index=? AND video_key=? AND batch_id IN
                (SELECT id FROM segmentation_batches WHERE profile_id=? AND dataset_id=?)""",
                (key[2], key[3], key[0], key[1]))
        revision = payload.expected_revision + 1
        connection.execute('''INSERT INTO segmentation_workspaces VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(profile_id,dataset_id,episode_index,video_key) DO UPDATE SET
            revision=excluded.revision, metadata_revision=excluded.metadata_revision,
            objects_json=excluded.objects_json, updated_at=excluded.updated_at''',
            (*key, revision, payload.metadata_revision, encoded, utc_now()))
    return {"revision": revision, "metadata_revision": payload.metadata_revision,
            "stale": False, "objects": json.loads(encoded)}
