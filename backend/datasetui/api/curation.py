"""Episode flags, annotations and curation recipes (runs, snapshots)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from datasetui.api.context import RouterContext
from datasetui.database import (
    AnnotationRevisionConflictError,
    DatasetNotFoundError,
    DatasetNotReadyError,
    DuplicateRecipeNameError,
    FlagRevisionConflictError,
    IdempotencyConflictError,
    ProfileNotFoundError,
    RecipeNotFoundError,
    RecipeRevisionMismatchError,
)
from datasetui.models import (
    CurationRecipe,
    CurationRecipeCreate,
    CurationRecipeSnapshot,
    CurationRecipeSnapshotCreate,
    CurationRecipeUpdate,
    CurationRunCreate,
    EpisodeAnnotations,
    EpisodeAnnotationsPut,
    EpisodeFlagPatch,
    EpisodeFlags,
    Job,
)

logger = logging.getLogger("datasetui.api")


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    dispatch_job = ctx.dispatch_job

    @router.get("/datasets/{dataset_id}/flags", response_model=EpisodeFlags)
    def get_episode_flags(dataset_id: str, profile_id: str) -> dict[str, Any]:
        try:
            return database.get_episode_flags(
                dataset_id=dataset_id, profile_id=profile_id
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc

    @router.patch("/datasets/{dataset_id}/flags", response_model=EpisodeFlags)
    def update_episode_flags(
        dataset_id: str, payload: EpisodeFlagPatch
    ) -> dict[str, Any]:
        try:
            return database.update_episode_flags(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                expected_revision=payload.expected_revision,
                changes=[change.model_dump() for change in payload.changes],
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc
        except FlagRevisionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Flags changed in another session. Reload and try again.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    @router.get(
        "/datasets/{dataset_id}/annotations/{episode_index}",
        response_model=EpisodeAnnotations,
    )
    def get_episode_annotations(
        dataset_id: str, episode_index: int, profile_id: str
    ) -> dict[str, Any]:
        try:
            return database.get_episode_annotations(
                dataset_id=dataset_id,
                profile_id=profile_id,
                episode_index=episode_index,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for annotation",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    @router.put(
        "/datasets/{dataset_id}/annotations/{episode_index}",
        response_model=EpisodeAnnotations,
    )
    def replace_episode_annotations(
        dataset_id: str, episode_index: int, payload: EpisodeAnnotationsPut
    ) -> dict[str, Any]:
        try:
            return database.replace_episode_annotations(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                episode_index=episode_index,
                expected_revision=payload.expected_revision,
                task_override=payload.task_override,
                atoms=[atom.model_dump(mode="json") for atom in payload.atoms],
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for annotation",
            ) from exc
        except AnnotationRevisionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Annotations changed in another session. Reload and try again.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    @router.get("/datasets/{dataset_id}/recipes", response_model=list[CurationRecipe])
    def list_curation_recipes(
        dataset_id: str,
        profile_id: str,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            return database.list_curation_recipes(
                dataset_id=dataset_id,
                profile_id=profile_id,
                include_archived=include_archived,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc

    @router.post(
        "/datasets/{dataset_id}/recipes",
        response_model=CurationRecipe,
        status_code=status.HTTP_201_CREATED,
    )
    def create_curation_recipe(
        dataset_id: str, payload: CurationRecipeCreate
    ) -> dict[str, Any]:
        try:
            return database.create_curation_recipe(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                name=payload.name,
                selection_mode=payload.selection_mode,
                operation=payload.operation,
                trim_config=payload.trim_config.model_dump(mode="json"),
                include_annotations=payload.include_annotations,
                relative_action=payload.relative_action.model_dump(mode="json"),
                split_config=payload.split_config.model_dump(mode="json"),
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc
        except DuplicateRecipeNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A recipe with this name already exists",
            ) from exc

    @router.patch("/recipes/{recipe_id}", response_model=CurationRecipe)
    def update_curation_recipe(
        recipe_id: str, payload: CurationRecipeUpdate
    ) -> dict[str, Any]:
        try:
            return database.update_curation_recipe(
                recipe_id,
                profile_id=payload.profile_id,
                name=payload.name,
                selection_mode=payload.selection_mode,
                operation=payload.operation,
                trim_config=(
                    payload.trim_config.model_dump(mode="json")
                    if payload.trim_config is not None
                    else None
                ),
                include_annotations=payload.include_annotations,
                relative_action=(
                    payload.relative_action.model_dump(mode="json")
                    if payload.relative_action is not None
                    else None
                ),
                split_config=(
                    payload.split_config.model_dump(mode="json")
                    if payload.split_config is not None
                    else None
                ),
                archived=payload.archived,
            )
        except (RecipeNotFoundError, ProfileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Recipe not found") from exc
        except DuplicateRecipeNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A recipe with this name already exists",
            ) from exc
        except (DatasetNotReadyError, RecipeRevisionMismatchError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The recipe belongs to a different dataset revision",
            ) from exc

    @router.post(
        "/recipes/{recipe_id}/runs",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def run_curation_recipe(
        recipe_id: str, payload: CurationRunCreate, response: Response
    ) -> dict[str, Any]:
        def matching_existing_job(existing: dict[str, Any]) -> dict[str, Any]:
            if existing["kind"] != "curation.materialize":
                raise IdempotencyConflictError(
                    "idempotency key is already bound to a different request"
                )
            existing_snapshot = database.get_curation_snapshot(
                existing["payload"].get("snapshot_id", "")
            )
            if (
                existing_snapshot["recipe_id"] != recipe_id
                or existing["payload"].get("output_name") != payload.output_name
            ):
                raise IdempotencyConflictError(
                    "idempotency key is already bound to a different request"
                )
            return existing

        try:
            existing = database.get_job_for_idempotency(
                payload.profile_id, payload.idempotency_key
            )
            if existing is not None:
                existing = matching_existing_job(existing)
                if existing["status"] == "queued":
                    existing = dispatch_job(existing)
                response.status_code = status.HTTP_200_OK
                return existing
            snapshot = database.snapshot_curation_recipe(
                recipe_id, profile_id=payload.profile_id
            )
            internal_payload = {
                "snapshot_id": snapshot["id"],
                "output_name": payload.output_name,
            }
            try:
                job, created = database.create_job(
                    kind="curation.materialize",
                    queue_name="cpu",
                    profile_id=payload.profile_id,
                    payload=internal_payload,
                    idempotency_key=payload.idempotency_key,
                )
            except IdempotencyConflictError:
                raced_job = database.get_job_for_idempotency(
                    payload.profile_id, payload.idempotency_key
                )
                if raced_job is None:
                    raise
                job = matching_existing_job(raced_job)
                created = False
            if created:
                database.record_curation_run(
                    job_id=job["id"],
                    snapshot_id=snapshot["id"],
                    output_name=payload.output_name,
                )
            elif job["status"] != "queued":
                response.status_code = status.HTTP_200_OK
                return job
            job = dispatch_job(job)
            if not created:
                response.status_code = status.HTTP_200_OK
            return job
        except (RecipeNotFoundError, ProfileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Recipe not found") from exc
        except (DatasetNotReadyError, RecipeRevisionMismatchError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The recipe belongs to a different dataset revision",
            ) from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch curation run")
            if "job" in locals():
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job queue unavailable",
            ) from exc

    @router.post(
        "/recipes/{recipe_id}/snapshots",
        response_model=CurationRecipeSnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    def snapshot_curation_recipe(
        recipe_id: str, payload: CurationRecipeSnapshotCreate
    ) -> dict[str, Any]:
        try:
            return database.snapshot_curation_recipe(
                recipe_id, profile_id=payload.profile_id
            )
        except (RecipeNotFoundError, ProfileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Recipe not found") from exc
        except (DatasetNotReadyError, RecipeRevisionMismatchError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The recipe belongs to a different dataset revision",
            ) from exc
