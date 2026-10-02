"""Schema migrations and small value helpers shared by the mixins."""

from __future__ import annotations

import json
import math
import random
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any


MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """
            CREATE TABLE profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT
            )
            """,
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                queue_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'queued', 'running', 'succeeded', 'failed',
                        'cancelled', 'interrupted'
                    )
                ),
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                payload_json TEXT NOT NULL,
                result_json TEXT,
                error_code TEXT,
                error_message TEXT,
                idempotency_key TEXT NOT NULL,
                rq_job_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                enqueued_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                UNIQUE (profile_id, idempotency_key)
            )
            """,
            """
            CREATE TABLE job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (job_id, sequence)
            )
            """,
            "CREATE INDEX jobs_status_created_idx ON jobs(status, created_at DESC)",
            "CREATE INDEX jobs_profile_created_idx ON jobs(profile_id, created_at DESC)",
            "CREATE INDEX job_events_job_idx ON job_events(job_id, sequence)",
        ),
    ),
    (
        2,
        (
            """
            CREATE TABLE datasets (
                id TEXT PRIMARY KEY,
                storage_area TEXT NOT NULL CHECK (storage_area IN ('raw', 'derived')),
                relative_path TEXT NOT NULL,
                name TEXT NOT NULL,
                codebase_version TEXT,
                readiness TEXT NOT NULL CHECK (
                    readiness IN ('ready', 'incomplete', 'unsupported', 'invalid')
                ),
                robot_type TEXT,
                total_episodes INTEGER,
                total_frames INTEGER,
                total_tasks INTEGER,
                fps REAL,
                fingerprint TEXT NOT NULL,
                info_mtime_ns INTEGER NOT NULL,
                info_size INTEGER NOT NULL,
                scan_error TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_scan_generation INTEGER NOT NULL,
                missing_since TEXT,
                UNIQUE (storage_area, relative_path)
            )
            """,
            """
            CREATE TABLE dataset_scan_generations (
                storage_area TEXT PRIMARY KEY CHECK (storage_area IN ('raw', 'derived')),
                generation INTEGER NOT NULL,
                started_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX datasets_library_idx ON datasets(missing_since, readiness, name)",
            "CREATE INDEX datasets_storage_idx ON datasets(storage_area, relative_path)",
        ),
    ),
    (
        3,
        (
            "ALTER TABLE jobs ADD COLUMN worker_id TEXT",
            "ALTER TABLE jobs ADD COLUMN heartbeat_at TEXT",
            "ALTER TABLE jobs ADD COLUMN lease_expires_at TEXT",
            "ALTER TABLE jobs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE jobs ADD COLUMN dispatch_generation INTEGER NOT NULL DEFAULT 0",
            "CREATE INDEX jobs_lease_idx ON jobs(status, lease_expires_at)",
            """
            CREATE TABLE hf_sources (
                repo_id TEXT PRIMARY KEY,
                desired_commit_sha TEXT NOT NULL,
                generation INTEGER NOT NULL,
                current_commit_sha TEXT,
                current_generation INTEGER,
                pointer_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (
                    pointer_confirmed IN (0, 1)
                ),
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE hf_import_requests (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                repo_id TEXT NOT NULL,
                requested_revision TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                generation INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE hf_revisions (
                repo_id TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                requested_revision TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                manifest_sha256 TEXT NOT NULL,
                file_count INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (repo_id, commit_sha)
            )
            """,
            "CREATE INDEX hf_import_requests_repo_idx ON hf_import_requests(repo_id, generation)",
            "CREATE INDEX hf_revisions_repo_idx ON hf_revisions(repo_id, imported_at DESC)",
        ),
    ),
    (
        4,
        (
            """
            CREATE TABLE flag_sets (
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset_id, dataset_fingerprint, profile_id)
            )
            """,
            """
            CREATE TABLE episode_flags (
                dataset_id TEXT NOT NULL,
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                episode_index INTEGER NOT NULL CHECK (episode_index >= 0),
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    dataset_id, dataset_fingerprint, profile_id, episode_index
                ),
                FOREIGN KEY (dataset_id, dataset_fingerprint, profile_id)
                    REFERENCES flag_sets(
                        dataset_id, dataset_fingerprint, profile_id
                    ) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE curation_recipes (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                selection_mode TEXT NOT NULL CHECK (
                    selection_mode IN ('all', 'flagged', 'unflagged')
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT,
                UNIQUE (
                    dataset_id, dataset_fingerprint, profile_id, name_key
                )
            )
            """,
            """
            CREATE TABLE curation_recipe_snapshots (
                id TEXT PRIMARY KEY,
                recipe_id TEXT NOT NULL REFERENCES curation_recipes(id),
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                recipe_name TEXT NOT NULL,
                selection_mode TEXT NOT NULL CHECK (
                    selection_mode IN ('all', 'flagged', 'unflagged')
                ),
                flag_revision INTEGER NOT NULL CHECK (flag_revision >= 0),
                flagged_episode_indices_json TEXT NOT NULL,
                selected_episode_indices_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX episode_flags_lookup_idx ON episode_flags(dataset_id, dataset_fingerprint, profile_id, episode_index)",
            "CREATE INDEX curation_recipes_lookup_idx ON curation_recipes(dataset_id, dataset_fingerprint, profile_id, archived_at, updated_at DESC)",
            "CREATE INDEX curation_recipe_snapshots_recipe_idx ON curation_recipe_snapshots(recipe_id, created_at DESC)",
        ),
    ),
    (
        5,
        (
            "ALTER TABLE curation_recipes ADD COLUMN operation TEXT NOT NULL DEFAULT 'subset'",
            'ALTER TABLE curation_recipes ADD COLUMN trim_config_json TEXT NOT NULL DEFAULT \'{"enabled":false,"threshold":0.02,"hold_time_s":0.5,"margin_s":1.0,"dimensions":[],"episode_overrides":{}}\'',
            "ALTER TABLE curation_recipe_snapshots ADD COLUMN operation TEXT NOT NULL DEFAULT 'subset'",
            'ALTER TABLE curation_recipe_snapshots ADD COLUMN trim_config_json TEXT NOT NULL DEFAULT \'{"enabled":false,"threshold":0.02,"hold_time_s":0.5,"margin_s":1.0,"dimensions":[],"episode_overrides":{}}\'',
            """
            CREATE TABLE curation_runs (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                snapshot_id TEXT NOT NULL REFERENCES curation_recipe_snapshots(id),
                output_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX curation_runs_snapshot_idx ON curation_runs(snapshot_id, created_at DESC)",
        ),
    ),
    (
        6,
        (
            "ALTER TABLE curation_recipes ADD COLUMN include_annotations INTEGER NOT NULL DEFAULT 0 CHECK (include_annotations IN (0, 1))",
            "ALTER TABLE curation_recipe_snapshots ADD COLUMN include_annotations INTEGER NOT NULL DEFAULT 0 CHECK (include_annotations IN (0, 1))",
            """
            CREATE TABLE episode_annotations (
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                episode_index INTEGER NOT NULL CHECK (episode_index >= 0),
                revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
                task_override TEXT,
                atoms_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    dataset_id, dataset_fingerprint, profile_id, episode_index
                )
            )
            """,
            """
            CREATE TABLE curation_annotation_snapshots (
                snapshot_id TEXT NOT NULL REFERENCES curation_recipe_snapshots(id)
                    ON DELETE CASCADE,
                episode_index INTEGER NOT NULL CHECK (episode_index >= 0),
                annotation_revision INTEGER NOT NULL CHECK (annotation_revision >= 0),
                task_override TEXT,
                atoms_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, episode_index)
            )
            """,
            "CREATE INDEX episode_annotations_lookup_idx ON episode_annotations(dataset_id, dataset_fingerprint, profile_id, updated_at DESC)",
        ),
    ),
    (
        7,
        (
            'ALTER TABLE curation_recipes ADD COLUMN relative_action_json TEXT NOT NULL DEFAULT \'{"enabled":false,"dimensions":[]}\'',
            'ALTER TABLE curation_recipe_snapshots ADD COLUMN relative_action_json TEXT NOT NULL DEFAULT \'{"enabled":false,"dimensions":[]}\'',
        ),
    ),
    (
        8,
        (
            """
            CREATE TABLE validation_runs (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('quick', 'full', 'export_gate')),
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX validation_runs_dataset_idx ON validation_runs(dataset_id, created_at DESC)",
        ),
    ),
    (
        9,
        ("ALTER TABLE validation_runs ADD COLUMN progress_json TEXT",),
    ),
    (
        10,
        (
            'ALTER TABLE curation_recipes ADD COLUMN split_config_json TEXT NOT NULL DEFAULT \'{"method":"flagged","eval_percent":20.0,"seed":0}\'',
            'ALTER TABLE curation_recipe_snapshots ADD COLUMN split_config_json TEXT NOT NULL DEFAULT \'{"method":"flagged","eval_percent":20.0,"seed":0}\'',
            "ALTER TABLE curation_recipe_snapshots ADD COLUMN eval_episode_indices_json TEXT NOT NULL DEFAULT '[]'",
        ),
    ),
    (
        11,
        ("ALTER TABLE validation_runs ADD COLUMN deleted_at TEXT",),
    ),
    (
        12,
        ("ALTER TABLE datasets ADD COLUMN display_name TEXT",),
    ),
    (
        13,
        ("ALTER TABLE jobs ADD COLUMN progress_json TEXT",),
    ),
    (
        14,
        (
            "ALTER TABLE jobs ADD COLUMN cancellation_requested_at TEXT",
            "ALTER TABLE jobs ADD COLUMN cancellation_guarded_at TEXT",
        ),
    ),
    (
        15,
        (
            """
            CREATE TABLE dataset_trash (
                dataset_id TEXT PRIMARY KEY REFERENCES datasets(id),
                storage_area TEXT NOT NULL CHECK (storage_area IN ('raw', 'derived')),
                original_relative_path TEXT NOT NULL,
                trash_relative_path TEXT NOT NULL,
                source_device INTEGER,
                source_inode INTEGER,
                state TEXT NOT NULL CHECK (
                    state IN ('moving', 'trashed', 'restoring', 'recovery_required')
                ),
                requested_by_profile_id TEXT NOT NULL REFERENCES profiles(id),
                requested_at TEXT NOT NULL,
                trashed_at TEXT,
                updated_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX dataset_trash_state_idx ON dataset_trash(state, updated_at)",
        ),
    ),
    (
        16,
        (
            "ALTER TABLE jobs ADD COLUMN validation_job_id TEXT REFERENCES jobs(id)",
            "ALTER TABLE jobs ADD COLUMN validation_ready INTEGER NOT NULL DEFAULT 1",
            "CREATE INDEX jobs_validation_idx ON jobs(validation_job_id, status)",
            """CREATE TABLE dataset_purge_reservations (
                dataset_id TEXT PRIMARY KEY REFERENCES datasets(id),
                job_id TEXT NOT NULL REFERENCES jobs(id),
                state TEXT NOT NULL CHECK (state IN ('reserved','purging','purged','failed')),
                updated_at TEXT NOT NULL
            )""",
        ),
    ),
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def utc_after(seconds: int) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=seconds))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def profile_name_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def recipe_name_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def random_eval_episode_indices(
    episode_indices: list[int], *, eval_percent: float, seed: int
) -> list[int]:
    """Choose round-half-up(N * percent / 100) episodes reproducibly."""
    count = math.floor(len(episode_indices) * eval_percent / 100 + 0.5)
    return sorted(random.Random(seed).sample(episode_indices, count))
