"""Episode flags, annotations, curation recipes and snapshots."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from datasetui.database.errors import (
    AnnotationRevisionConflictError,
    DatasetNotFoundError,
    DatasetNotReadyError,
    DuplicateRecipeNameError,
    FlagRevisionConflictError,
    ProfileNotFoundError,
    RecipeNotFoundError,
    RecipeRevisionMismatchError,
)
from datasetui.database.schema import (
    json_dump,
    random_eval_episode_indices,
    recipe_name_key,
    utc_now,
)


class CurationMixin:
    """Episode flags, annotations, curation recipes and snapshots."""

    @staticmethod
    def _curation_source(
        connection: sqlite3.Connection,
        *,
        dataset_id: str,
        profile_id: str,
    ) -> sqlite3.Row:
        profile = connection.execute(
            "SELECT id FROM profiles WHERE id = ? AND archived_at IS NULL",
            (profile_id,),
        ).fetchone()
        if profile is None:
            raise ProfileNotFoundError(profile_id)

        dataset = connection.execute(
            """
            SELECT id, fingerprint, total_episodes, readiness, missing_since
            FROM datasets WHERE id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM dataset_trash t WHERE t.dataset_id = datasets.id
              )
            """,
            (dataset_id,),
        ).fetchone()
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        if (
            dataset["missing_since"] is not None
            or dataset["readiness"] != "ready"
            or dataset["total_episodes"] is None
        ):
            raise DatasetNotReadyError(dataset_id)
        return dataset

    @staticmethod
    def _episode_flags_record(
        connection: sqlite3.Connection,
        *,
        dataset_id: str,
        dataset_fingerprint: str,
        profile_id: str,
    ) -> dict[str, Any]:
        flag_set = connection.execute(
            """
            SELECT revision, updated_at FROM flag_sets
            WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
            """,
            (dataset_id, dataset_fingerprint, profile_id),
        ).fetchone()
        indices = [
            row["episode_index"]
            for row in connection.execute(
                """
                SELECT episode_index FROM episode_flags
                WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
                ORDER BY episode_index
                """,
                (dataset_id, dataset_fingerprint, profile_id),
            )
        ]
        return {
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "profile_id": profile_id,
            "revision": flag_set["revision"] if flag_set else 0,
            "episode_indices": indices,
            "updated_at": flag_set["updated_at"] if flag_set else None,
        }

    def get_episode_flags(self, *, dataset_id: str, profile_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            return self._episode_flags_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
            )

    def update_episode_flags(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        expected_revision: int,
        changes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            total_episodes = dataset["total_episodes"]
            for change in changes:
                if change["episode_index"] >= total_episodes:
                    raise ValueError("episode index is outside the dataset")

            fingerprint = dataset["fingerprint"]
            flag_set = connection.execute(
                """
                SELECT revision FROM flag_sets
                WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
                """,
                (dataset_id, fingerprint, profile_id),
            ).fetchone()
            current_revision = flag_set["revision"] if flag_set else 0
            if current_revision != expected_revision:
                raise FlagRevisionConflictError(str(current_revision))

            existing = {
                row["episode_index"]
                for row in connection.execute(
                    """
                    SELECT episode_index FROM episode_flags
                    WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
                    """,
                    (dataset_id, fingerprint, profile_id),
                )
            }
            effective = [
                change
                for change in changes
                if (change["episode_index"] in existing) != change["flagged"]
            ]
            if effective:
                connection.execute(
                    """
                    INSERT INTO flag_sets(
                        dataset_id, dataset_fingerprint, profile_id,
                        revision, updated_at
                    ) VALUES (?, ?, ?, 0, ?)
                    ON CONFLICT(dataset_id, dataset_fingerprint, profile_id)
                    DO NOTHING
                    """,
                    (dataset_id, fingerprint, profile_id, now),
                )
                for change in effective:
                    key = (
                        dataset_id,
                        fingerprint,
                        profile_id,
                        change["episode_index"],
                    )
                    if change["flagged"]:
                        connection.execute(
                            """
                            INSERT INTO episode_flags(
                                dataset_id, dataset_fingerprint, profile_id,
                                episode_index, created_at
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (*key, now),
                        )
                    else:
                        connection.execute(
                            """
                            DELETE FROM episode_flags
                            WHERE dataset_id = ? AND dataset_fingerprint = ?
                              AND profile_id = ? AND episode_index = ?
                            """,
                            key,
                        )
                connection.execute(
                    """
                    UPDATE flag_sets SET revision = revision + 1, updated_at = ?
                    WHERE dataset_id = ? AND dataset_fingerprint = ?
                      AND profile_id = ?
                    """,
                    (now, dataset_id, fingerprint, profile_id),
                )

            return self._episode_flags_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint,
                profile_id=profile_id,
            )

    @staticmethod
    def _episode_annotations_record(
        connection: sqlite3.Connection,
        *,
        dataset_id: str,
        dataset_fingerprint: str,
        profile_id: str,
        episode_index: int,
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT revision, task_override, atoms_json, updated_at
            FROM episode_annotations
            WHERE dataset_id = ? AND dataset_fingerprint = ?
              AND profile_id = ? AND episode_index = ?
            """,
            (dataset_id, dataset_fingerprint, profile_id, episode_index),
        ).fetchone()
        return {
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "profile_id": profile_id,
            "episode_index": episode_index,
            "revision": row["revision"] if row else 0,
            "task_override": row["task_override"] if row else None,
            "atoms": json.loads(row["atoms_json"]) if row else [],
            "updated_at": row["updated_at"] if row else None,
        }

    def get_episode_annotations(
        self, *, dataset_id: str, profile_id: str, episode_index: int
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            if episode_index < 0 or episode_index >= dataset["total_episodes"]:
                raise ValueError("episode index is outside the dataset")
            return self._episode_annotations_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
                episode_index=episode_index,
            )

    def replace_episode_annotations(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        episode_index: int,
        expected_revision: int,
        task_override: str | None,
        atoms: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            if episode_index < 0 or episode_index >= dataset["total_episodes"]:
                raise ValueError("episode index is outside the dataset")
            current = self._episode_annotations_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
                episode_index=episode_index,
            )
            if current["revision"] != expected_revision:
                raise AnnotationRevisionConflictError(dataset_id)
            connection.execute(
                """
                INSERT INTO episode_annotations(
                    dataset_id, dataset_fingerprint, profile_id, episode_index,
                    revision, task_override, atoms_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(
                    dataset_id, dataset_fingerprint, profile_id, episode_index
                ) DO UPDATE SET
                    revision = episode_annotations.revision + 1,
                    task_override = excluded.task_override,
                    atoms_json = excluded.atoms_json,
                    updated_at = excluded.updated_at
                """,
                (
                    dataset_id,
                    dataset["fingerprint"],
                    profile_id,
                    episode_index,
                    task_override,
                    json_dump(atoms),
                    now,
                    now,
                ),
            )
            return self._episode_annotations_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
                episode_index=episode_index,
            )

    def create_curation_recipe(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        name: str,
        selection_mode: str,
        operation: str = "subset",
        trim_config: dict[str, Any] | None = None,
        include_annotations: bool = False,
        relative_action: dict[str, Any] | None = None,
        split_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        recipe_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                dataset = self._curation_source(
                    connection, dataset_id=dataset_id, profile_id=profile_id
                )
                connection.execute(
                    """
                    INSERT INTO curation_recipes(
                        id, dataset_id, dataset_fingerprint, profile_id,
                        name, name_key, selection_mode, operation,
                        trim_config_json, include_annotations, relative_action_json,
                        split_config_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recipe_id,
                        dataset_id,
                        dataset["fingerprint"],
                        profile_id,
                        name,
                        recipe_name_key(name),
                        selection_mode,
                        operation,
                        json_dump(trim_config or {"enabled": False}),
                        int(include_annotations),
                        json_dump(
                            relative_action or {"enabled": False, "dimensions": []}
                        ),
                        json_dump(
                            split_config
                            or {"method": "flagged", "eval_percent": 20.0, "seed": 0}
                        ),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "curation_recipes.dataset_id" in str(exc):
                raise DuplicateRecipeNameError(name) from exc
            raise
        return self.get_curation_recipe(recipe_id, profile_id=profile_id)

    def list_curation_recipes(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            archived_clause = "" if include_archived else "AND archived_at IS NULL"
            rows = connection.execute(
                f"""
                SELECT id, dataset_id, dataset_fingerprint, profile_id, name,
                       selection_mode, operation, trim_config_json,
                       include_annotations,
                       relative_action_json,
                       split_config_json,
                       created_at, updated_at, archived_at
                FROM curation_recipes
                WHERE dataset_id = ? AND dataset_fingerprint = ?
                  AND profile_id = ? {archived_clause}
                ORDER BY updated_at DESC, name_key
                """,
                (dataset_id, dataset["fingerprint"], profile_id),
            ).fetchall()
        return [self._decode_recipe(row) for row in rows]

    def get_curation_recipe(self, recipe_id: str, *, profile_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, dataset_id, dataset_fingerprint, profile_id, name,
                       selection_mode, operation, trim_config_json,
                       include_annotations,
                       relative_action_json,
                       split_config_json,
                       created_at, updated_at, archived_at
                FROM curation_recipes WHERE id = ? AND profile_id = ?
                """,
                (recipe_id, profile_id),
            ).fetchone()
        if row is None:
            raise RecipeNotFoundError(recipe_id)
        return self._decode_recipe(row)

    def update_curation_recipe(
        self,
        recipe_id: str,
        *,
        profile_id: str,
        name: str | None,
        selection_mode: str | None,
        operation: str | None = None,
        trim_config: dict[str, Any] | None = None,
        include_annotations: bool | None = None,
        relative_action: dict[str, Any] | None = None,
        split_config: dict[str, Any] | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                recipe = connection.execute(
                    "SELECT * FROM curation_recipes WHERE id = ? AND profile_id = ?",
                    (recipe_id, profile_id),
                ).fetchone()
                if recipe is None:
                    raise RecipeNotFoundError(recipe_id)
                dataset = self._curation_source(
                    connection,
                    dataset_id=recipe["dataset_id"],
                    profile_id=profile_id,
                )
                if dataset["fingerprint"] != recipe["dataset_fingerprint"]:
                    raise RecipeRevisionMismatchError(recipe_id)
                next_name = name if name is not None else recipe["name"]
                next_selection = (
                    selection_mode
                    if selection_mode is not None
                    else recipe["selection_mode"]
                )
                next_operation = (
                    operation if operation is not None else recipe["operation"]
                )
                next_trim_config = (
                    json_dump(trim_config)
                    if trim_config is not None
                    else recipe["trim_config_json"]
                )
                next_include_annotations = (
                    int(include_annotations)
                    if include_annotations is not None
                    else recipe["include_annotations"]
                )
                next_relative_action = (
                    json_dump(relative_action)
                    if relative_action is not None
                    else recipe["relative_action_json"]
                )
                next_split_config = (
                    json_dump(split_config)
                    if split_config is not None
                    else recipe["split_config_json"]
                )
                next_archived_at = recipe["archived_at"]
                if archived is True:
                    next_archived_at = now
                elif archived is False:
                    next_archived_at = None
                connection.execute(
                    """
                    UPDATE curation_recipes
                    SET name = ?, name_key = ?, selection_mode = ?, operation = ?,
                        trim_config_json = ?, include_annotations = ?, relative_action_json = ?,
                        split_config_json = ?,
                        archived_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        next_name,
                        recipe_name_key(next_name),
                        next_selection,
                        next_operation,
                        next_trim_config,
                        next_include_annotations,
                        next_relative_action,
                        next_split_config,
                        next_archived_at,
                        now,
                        recipe_id,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "curation_recipes.dataset_id" in str(exc):
                raise DuplicateRecipeNameError(name or "") from exc
            raise
        return self.get_curation_recipe(recipe_id, profile_id=profile_id)

    def snapshot_curation_recipe(
        self, recipe_id: str, *, profile_id: str
    ) -> dict[str, Any]:
        snapshot_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            recipe = connection.execute(
                """
                SELECT * FROM curation_recipes
                WHERE id = ? AND profile_id = ? AND archived_at IS NULL
                """,
                (recipe_id, profile_id),
            ).fetchone()
            if recipe is None:
                raise RecipeNotFoundError(recipe_id)
            dataset = self._curation_source(
                connection,
                dataset_id=recipe["dataset_id"],
                profile_id=profile_id,
            )
            if dataset["fingerprint"] != recipe["dataset_fingerprint"]:
                raise RecipeRevisionMismatchError(recipe_id)
            flags = self._episode_flags_record(
                connection,
                dataset_id=recipe["dataset_id"],
                dataset_fingerprint=recipe["dataset_fingerprint"],
                profile_id=profile_id,
            )
            flagged = flags["episode_indices"]
            flagged_set = set(flagged)
            if recipe["operation"] == "delete_flagged":
                selected = [
                    index
                    for index in range(dataset["total_episodes"])
                    if index not in flagged_set
                ]
            elif recipe["selection_mode"] == "all":
                selected = list(range(dataset["total_episodes"]))
            elif recipe["selection_mode"] == "flagged":
                selected = flagged
            else:
                selected = [
                    index
                    for index in range(dataset["total_episodes"])
                    if index not in flagged_set
                ]
            split_config = json.loads(recipe["split_config_json"])
            if (
                recipe["operation"] == "train_eval_split"
                and split_config["method"] == "flagged"
            ):
                selected = list(range(dataset["total_episodes"]))
            if not selected:
                raise ValueError("recipe selection cannot be empty")
            eval_episode_indices: list[int] = []
            if recipe["operation"] == "train_eval_split":
                if split_config["method"] == "random":
                    eval_episode_indices = random_eval_episode_indices(
                        selected,
                        eval_percent=split_config["eval_percent"],
                        seed=split_config["seed"],
                    )
                else:
                    eval_episode_indices = flagged
                    if not flagged or len(flagged) == dataset["total_episodes"]:
                        raise ValueError(
                            "flag-based train/eval split requires non-empty train and eval sets"
                        )
            connection.execute(
                """
                INSERT INTO curation_recipe_snapshots(
                    id, recipe_id, dataset_id, dataset_fingerprint, profile_id,
                    recipe_name, selection_mode, flag_revision, operation,
                    trim_config_json, include_annotations, relative_action_json,
                    split_config_json, eval_episode_indices_json,
                    flagged_episode_indices_json, selected_episode_indices_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    recipe_id,
                    recipe["dataset_id"],
                    recipe["dataset_fingerprint"],
                    profile_id,
                    recipe["name"],
                    recipe["selection_mode"],
                    flags["revision"],
                    recipe["operation"],
                    recipe["trim_config_json"],
                    recipe["include_annotations"],
                    recipe["relative_action_json"],
                    recipe["split_config_json"],
                    json_dump(eval_episode_indices),
                    json_dump(flagged),
                    json_dump(selected),
                    now,
                ),
            )
            annotation_episode_indices: list[int] = []
            if recipe["include_annotations"]:
                selected_set = set(selected)
                annotation_rows = connection.execute(
                    """
                    SELECT episode_index, revision, task_override, atoms_json
                    FROM episode_annotations
                    WHERE dataset_id = ? AND dataset_fingerprint = ?
                      AND profile_id = ?
                    ORDER BY episode_index
                    """,
                    (
                        recipe["dataset_id"],
                        recipe["dataset_fingerprint"],
                        profile_id,
                    ),
                ).fetchall()
                for annotation in annotation_rows:
                    if annotation["episode_index"] not in selected_set:
                        continue
                    connection.execute(
                        """
                        INSERT INTO curation_annotation_snapshots(
                            snapshot_id, episode_index, annotation_revision,
                            task_override, atoms_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            annotation["episode_index"],
                            annotation["revision"],
                            annotation["task_override"],
                            annotation["atoms_json"],
                        ),
                    )
                    annotation_episode_indices.append(annotation["episode_index"])
        return {
            "id": snapshot_id,
            "recipe_id": recipe_id,
            "dataset_id": recipe["dataset_id"],
            "dataset_fingerprint": recipe["dataset_fingerprint"],
            "profile_id": profile_id,
            "recipe_name": recipe["name"],
            "selection_mode": recipe["selection_mode"],
            "operation": recipe["operation"],
            "trim_config": json.loads(recipe["trim_config_json"]),
            "include_annotations": bool(recipe["include_annotations"]),
            "relative_action": json.loads(recipe["relative_action_json"]),
            "split_config": split_config,
            "annotation_episode_indices": annotation_episode_indices,
            "flag_revision": flags["revision"],
            "flagged_episode_indices": flagged,
            "selected_episode_indices": selected,
            "eval_episode_indices": eval_episode_indices,
            "created_at": now,
        }

    def get_curation_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT s.*, d.storage_area, d.relative_path, d.name AS dataset_name,
                       d.codebase_version, d.readiness, d.missing_since, d.fingerprint
                FROM curation_recipe_snapshots s
                JOIN datasets d ON d.id = s.dataset_id
                WHERE s.id = ?
                """,
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise RecipeNotFoundError(snapshot_id)
        result = dict(row)
        result["flagged_episode_indices"] = json.loads(
            result.pop("flagged_episode_indices_json")
        )
        result["selected_episode_indices"] = json.loads(
            result.pop("selected_episode_indices_json")
        )
        result["eval_episode_indices"] = json.loads(
            result.pop("eval_episode_indices_json")
        )
        result["trim_config"] = json.loads(result.pop("trim_config_json"))
        result["include_annotations"] = bool(result["include_annotations"])
        result["relative_action"] = json.loads(result.pop("relative_action_json"))
        result["split_config"] = json.loads(result.pop("split_config_json"))
        if (
            result["operation"] == "train_eval_split"
            and result["split_config"]["method"] == "flagged"
            and not result["eval_episode_indices"]
        ):
            result["eval_episode_indices"] = list(result["flagged_episode_indices"])
        with self.connect() as connection:
            result["annotation_episode_indices"] = [
                annotation["episode_index"]
                for annotation in connection.execute(
                    """
                    SELECT episode_index FROM curation_annotation_snapshots
                    WHERE snapshot_id = ? ORDER BY episode_index
                    """,
                    (snapshot_id,),
                )
            ]
        return result

    def get_curation_snapshot_annotations(
        self, snapshot_id: str
    ) -> dict[int, dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT episode_index, annotation_revision, task_override, atoms_json
                FROM curation_annotation_snapshots
                WHERE snapshot_id = ? ORDER BY episode_index
                """,
                (snapshot_id,),
            ).fetchall()
        return {
            row["episode_index"]: {
                "revision": row["annotation_revision"],
                "task_override": row["task_override"],
                "atoms": json.loads(row["atoms_json"]),
            }
            for row in rows
        }

    def record_curation_run(
        self, *, job_id: str, snapshot_id: str, output_name: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO curation_runs(
                    job_id, snapshot_id, output_name, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (job_id, snapshot_id, output_name, utc_now()),
            )
