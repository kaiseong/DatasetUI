from __future__ import annotations

import hashlib
import json
import fcntl
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from datasetui.config import Settings
from datasetui.content_integrity import (
    ContentIntegrityError,
    dataset_content_fingerprint,
    dataset_content_manifest,
)
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.datasets import inspect_dataset, scan_storage_area
from datasetui.job_progress import JobProgressReporter
from datasetui.segmentation_contract import SegmentationSpec, decode_background
from datasetui.segmentation_selection import (
    brush_hints,
    has_keep_objects,
    retained_mask,
    validate_detections,
)
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import (
    _DatasetSource,
    _read_regular_bytes,
    _report_progress,
    _safe_dataset_root,
    _write_json_atomic,
)
from datasetui.validation import validate_dataset_root


MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MASK_TARGETS = ("replace", "protect")
MAX_IMAGE_PIXELS = 16_000_000


class SegmentationError(CurationTransformError):
    pass


def load_source(
    database: Database,
    settings: Settings,
    dataset_id: str,
    fingerprint: str | None = None,
    *,
    verify_content: bool = True,
) -> tuple[Path, _DatasetSource]:
    record = database.get_dataset(str(dataset_id))
    # Segmentation approvals bind full content independently of older registry
    # deployments whose fingerprint intentionally covers only meta/info.json.
    expected = fingerprint
    if not record["available"] or record["readiness"] != "ready":
        raise RecipeRevisionMismatchError(str(dataset_id))
    root = _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    if verify_content:
        try:
            actual = dataset_content_fingerprint(root, reuse_file_digests=True)
        except ContentIntegrityError as exc:
            raise RecipeRevisionMismatchError(str(dataset_id)) from exc
        if expected is not None and actual != expected:
            raise RecipeRevisionMismatchError(str(dataset_id))
    try:
        info_path = _safe_regular_path(root, root / "meta/info.json")
        _safe_directory_path(root, root / "data")
        info = json.loads(_read_regular_bytes(info_path, max_bytes=2 * 1024 * 1024))
        if not isinstance(info, dict):
            raise ValueError
        source = _DatasetSource(root, info)
        if verify_content:
            source.segmentation_fingerprint = actual
    except Exception as exc:
        raise SegmentationError("Dataset source could not be loaded") from exc
    return root, source


def dataset_scope(
    database: Database, settings: Settings, dataset_id: str
) -> dict[str, Any]:
    _, source = load_source(database, settings, dataset_id)
    episodes = []
    for episode_index in range(int(source.info["total_episodes"])):
        data, _ = source.episode(episode_index)
        episodes.append({"episode_index": episode_index, "length": len(data)})
    return {
        "fingerprint": source.segmentation_fingerprint,
        "video_keys": list(source.video_keys),
        "episodes": episodes,
    }


def source_frame(
    database: Database,
    settings: Settings,
    dataset_id: str,
    episode_index: int,
    video_key: str,
    frame_index: int,
) -> bytes:
    root, source = load_source(database, settings, dataset_id, verify_content=False)
    data, metadata = _episode(source, episode_index)
    _validate_video_selection(source, video_key, frame_index, len(data))
    path, start = source.video_source(episode_index, video_key, metadata)
    path = _safe_regular_path(root, path)
    frame = _decode_one(path, start + frame_index)
    if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
        raise SegmentationError("Video frame exceeds the image pixel limit")
    image = Image.fromarray(frame, mode="RGB")
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG", compress_level=6)
    return output.getvalue()


def create_preview(
    database: Database,
    settings: Settings,
    *,
    job_id: str,
    worker_id: str,
    spec: dict[str, Any],
    engine: Any = None,
) -> dict[str, Any]:
    progress = JobProgressReporter(
        database,
        job_id=job_id,
        worker_id=worker_id,
    )
    _report_progress(
        progress,
        stage="preparing",
        completed=0,
        total=1,
        unit="items",
        current_item="미리보기 원본 확인",
        force=True,
    )
    from datasetui.segmentation_pending import resolve_preview_spec

    spec = resolve_preview_spec(database, settings, spec)
    parsed = SegmentationSpec.model_validate(spec)
    from datasetui.segmentation_candidates import validate_candidate_references
    if any(p.selected_candidates for p in parsed.prompts):
        validate_candidate_references(database, settings, parsed, database.get_job(job_id)["profile_id"])
    normalized = parsed.model_dump(mode="json")
    dataset_id = str(parsed.dataset_id)
    root, source = load_source(database, settings, dataset_id, parsed.fingerprint)
    data, metadata = _episode(source, parsed.episode_index)
    frame_count = len(data)
    maximum = int(getattr(settings, "segmentation_max_frames", 3600))
    if frame_count < 1 or frame_count > maximum:
        raise SegmentationError("Episode exceeds the segmentation frame limit")
    _validate_video_selection(source, parsed.video_key, 0, frame_count)
    for prompt in parsed.prompts:
        if prompt.frame_index >= frame_count:
            raise SegmentationError("Prompt frame is outside the episode")
    for correction in parsed.corrections:
        if correction.frame_index >= frame_count:
            raise SegmentationError("Correction frame is outside the episode")
    from datasetui.job_progress import WeightedProgress

    # Whole-job progress: stages weighted by expected per-frame cost.
    progress = WeightedProgress(
        progress,
        {
            "preparing": frame_count * 0.3,
            "read": 0 if parsed.source_preview_id else frame_count,
            "segment": 0
            if parsed.source_preview_id or not parsed.prompts
            else frame_count * 4 * _estimated_sam_passes(parsed),
            "write": frame_count * 1.5,
            "validate": frame_count * 0.3,
            "publish": frame_count * 0.1,
        },
    )

    source_path, source_start = source.video_source(
        parsed.episode_index, parsed.video_key, metadata
    )
    source_path = _safe_regular_path(root, source_path)
    probe = _decode_one(source_path, source_start)
    height, width = probe.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise SegmentationError("Video frame exceeds the image pixel limit")
    background = (
        decode_background(parsed.background_base64).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        if parsed.render_mode == "image"
        else Image.new("RGB", (width, height), "black")
    )
    background_bytes = _png_bytes(background)
    background_sha256 = hashlib.sha256(background_bytes).hexdigest()
    stored_spec = {**normalized, "background_sha256": background_sha256}
    stored_spec.pop("background_base64", None)
    recipe_hash = _canonical_hash(stored_spec)
    _report_progress(
        progress,
        stage="preparing",
        completed=1,
        total=1,
        unit="items",
        current_item="미리보기 원본 확인",
    )
    preview_parent = settings.jobs_root / "segmentation"
    preview_parent.mkdir(parents=True, exist_ok=True)
    final = preview_parent / job_id
    if final.exists():
        database.assert_job_lease(job_id, worker_id=worker_id)
        result = _reuse_preview(final, recipe_hash, parsed.fingerprint, job_id)
        _report_progress(
            progress,
            stage="complete",
            completed=1,
            total=1,
            unit="items",
            current_item="기존 미리보기 재사용",
            force=True,
        )
        return result

    staging = Path(tempfile.mkdtemp(prefix=f".{job_id}-", dir=preview_parent))
    try:

        def lease() -> None:
            database.assert_job_lease(job_id, worker_id=worker_id)

        lease()
        from datasetui.segmentation_masks import (
            apply_object_corrections,
            apply_protected_regions,
            apply_selection,
            mask_inputs,
            reuse_masks,
            review_signals,
        )

        cached_provenance = None
        if parsed.source_preview_id:
            cached_provenance = reuse_masks(
                database,
                settings,
                parsed,
                staging,
                owner_profile_id=database.get_job(job_id)["profile_id"],
            )
            clip_width, clip_height = width, height
        else:
            clip_width, clip_height = _write_episode_clip(
                source_path,
                source_start,
                frame_count,
                source.fps,
                staging / "original.mp4",
                lease,
                on_progress=progress,
                current_item=f"에피소드 {parsed.episode_index} · 카메라 {parsed.video_key}",
            )
        if (clip_width, clip_height) != (width, height):
            raise SegmentationError("Video dimensions changed while creating preview")
        background.save(staging / "background.png", format="PNG")
        prompt_payload = [item.model_dump(mode="json") for item in parsed.prompts]
        if not prompt_payload and cached_provenance is None:
            for target in MASK_TARGETS:
                (staging / target).mkdir()
                for index in range(frame_count):
                    Image.new("L", (width, height), 0).save(
                        staging / target / f"{index:06d}.png"
                    )
            cached_provenance = {
                "engine": "manual-protection",
                "candidates": [],
                "frame_count": frame_count,
            }
        for correction in parsed.corrections:
            if correction.object_id is not None:
                # Cover the WHOLE stroke with bounded SAM seeds. The full stroke
                # and radius are rasterized exactly at its keyframe below.
                seeds = [
                    correction.points[int(index)]
                    for index in np.linspace(
                        0,
                        len(correction.points) - 1,
                        min(64, len(correction.points)),
                        dtype=int,
                    )
                ]
                prompt_payload.append(
                    {
                        "object_id": correction.object_id,
                        "member_candidate_id": correction.member_candidate_id,
                        "frame_index": correction.frame_index,
                        "target": correction.target,
                        "points": brush_hints(correction, width, height)
                        if parsed.mode == "object_selection"
                        else [
                            {
                                "x": p.x,
                                "y": p.y,
                                "label": 1 if correction.operation == "add" else 0,
                            }
                            for p in seeds
                        ],
                    }
                )
        _report_progress(
            progress,
            stage="segment",
            completed=0,
            total=0,
            unit="items",
            current_item="SAM 3.1 마스크 생성",
            force=True,
        )
        segmenter = engine or _default_engine(
            settings, mixed_spatial=parsed.mode == "object_selection"
        )
        provenance = cached_provenance or segmenter.propagate(
            video_path=staging / "original.mp4",
            prompts=prompt_payload,
            frame_count=frame_count,
            output_dir=staging,
            check_lease=lease,
            **_engine_progress(segmenter, progress),
        )
        if not isinstance(provenance, dict):
            raise SegmentationError("Segmentation engine returned invalid provenance")
        validate_detections(parsed, provenance)
        provenance["correction_policy"] = (
            "radius-sam-hints-no-raster-v1"
            if parsed.mode == "object_selection"
            else "same-track-whole-stroke-seeds-exact-keyframe-raster-v1"
        )
        apply_object_corrections(
            staging, parsed, provenance, frame_count, width, height
        )
        selection_required = apply_selection(
            staging, parsed, provenance, frame_count, width, height
        )
        _report_progress(
            progress,
            stage="segment",
            completed=1,
            total=1,
            unit="items",
            current_item="SAM 3.1 마스크 생성",
            force=True,
        )
        _apply_corrections(
            staging,
            parsed,
            frame_count,
            width,
            height,
            include_object_corrections="candidates" not in provenance,
            on_progress=progress,
        )
        apply_protected_regions(staging, parsed, frame_count, width, height)
        signals = review_signals(staging, parsed, frame_count, width, height)
        review_blocked = (
            sum(item["reason"] == "empty_mask" for item in signals) == frame_count
        )
        from datasetui.segmentation_masks import (
            coverage_signals,
            object_coverage,
            selection_review_signals,
        )

        coverage = (
            []
            if selection_required
            else object_coverage(provenance, parsed.selected_candidate_ids, frame_count)
        )
        signals = (
            selection_review_signals(provenance) + coverage_signals(coverage) + signals
        )
        _render_preview(
            staging,
            parsed.mode,
            background,
            frame_count,
            source.fps,
            width,
            height,
            lease,
            on_progress=progress,
            has_keep=has_keep_objects(parsed),
        )
        _report_progress(
            progress,
            stage="validate",
            completed=0,
            total=1,
            unit="items",
            current_item="미리보기 결과 검사",
            force=True,
        )
        lease()
        if dataset_content_fingerprint(root, reuse_file_digests=True) != parsed.fingerprint:
            raise RecipeRevisionMismatchError(dataset_id)
        manifest = {
            "schema_version": 1,
            "preview_id": job_id,
            "recipe_hash": recipe_hash,
            "fingerprint": parsed.fingerprint,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": source.fps,
            "spec": stored_spec,
            "source": {
                "dataset_id": dataset_id,
                "video_path": source_path.relative_to(root).as_posix(),
                "start_frame": source_start,
                "total_frames": _video_frame_count(source_path),
            },
            "model": provenance,
            "mask_input_hash": _canonical_hash(mask_inputs(normalized)),
            "selection_required": selection_required,
            "review_signals": signals,
            "review_blocked": review_blocked,
            "object_coverage": coverage,
            "artifacts": _preview_artifact_manifest(staging),
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        _report_progress(
            progress,
            stage="validate",
            completed=1,
            total=1,
            unit="items",
            current_item="미리보기 결과 검사",
        )
        lease()
        _report_progress(
            progress,
            stage="publish",
            completed=0,
            total=1,
            unit="items",
            current_item="미리보기 게시",
            force=True,
        )
        with _output_lock(settings, f"preview-{job_id}"):
            if final.exists() or final.is_symlink():
                result = _reuse_preview(final, recipe_hash, parsed.fingerprint, job_id)
                _report_progress(
                    progress,
                    stage="complete",
                    completed=1,
                    total=1,
                    unit="items",
                    current_item="기존 미리보기 재사용",
                    force=True,
                )
                return result
            database.begin_job_finalization(job_id, worker_id=worker_id)
            staging.rename(final)
        artifact_fingerprint = dataset_content_fingerprint(final, reuse_file_digests=True)
        result = _preview_result(manifest, artifact_fingerprint)
        _report_progress(
            progress,
            stage="publish",
            completed=1,
            total=1,
            unit="items",
            current_item="미리보기 게시",
        )
        _report_progress(
            progress,
            stage="complete",
            completed=1,
            total=1,
            unit="items",
            current_item="미리보기 완료",
            force=True,
        )
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def export_preview(
    database: Database,
    settings: Settings,
    *,
    job_id: str,
    worker_id: str,
    preview_id: str | None = None,
    output_name: str,
    preview_ids: list[str] | None = None,
    previews: list[dict[str, Any]] | None = None,
    recompute_statistics: bool = False,
) -> dict[str, Any]:
    from datasetui.segmentation_api import verified_preview

    progress = JobProgressReporter(
        database,
        job_id=job_id,
        worker_id=worker_id,
        output_name=output_name,
    )
    _report_progress(
        progress,
        stage="preparing",
        completed=0,
        total=1,
        unit="items",
        current_item="승인된 미리보기 확인",
        force=True,
    )

    selected_ids = (
        preview_ids
        or [str(item["preview_id"]) for item in previews or []]
        or ([str(preview_id)] if preview_id else [])
    )
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise SegmentationError("Select distinct approved previews for export")
    selections = []
    for selected_id in selected_ids:
        try:
            _, expected_result = verified_preview(
                database, settings, selected_id, verify_source=False
            )
        except (ValueError, ContentIntegrityError) as exc:
            raise SegmentationError("Approved preview is stale") from exc
        preview_root = settings.jobs_root / "segmentation" / selected_id
        manifest = _read_manifest(preview_root)
        for key in ("recipe_hash", "fingerprint"):
            if expected_result.get(key) != manifest.get(key):
                raise SegmentationError("Approved preview metadata is stale")
        artifact_fingerprint = dataset_content_fingerprint(preview_root, reuse_file_digests=True)
        if artifact_fingerprint != expected_result["artifact_fingerprint"]:
            raise SegmentationError("Approved preview artifacts were modified")
        if manifest.get("selection_required"):
            raise SegmentationError("Select text candidates before exporting")
        if manifest.get("review_blocked"):
            raise SegmentationError(
                "The complete clip has an empty selected mask; correct the prompts"
            )
        selections.append((preview_root, manifest, artifact_fingerprint))
    preview_root, manifest, _ = selections[0]
    if (
        len(
            {
                (item[1]["source"]["dataset_id"], item[1]["fingerprint"])
                for item in selections
            }
        )
        != 1
    ):
        raise SegmentationError(
            "All previews must belong to the same source dataset revision"
        )
    view_keys = [
        (item[1]["spec"]["episode_index"], item[1]["spec"]["video_key"])
        for item in selections
    ]
    if len(view_keys) != len(set(view_keys)):
        raise SegmentationError(
            "Only one approved preview is allowed for each episode and camera"
        )
    preview_id = selected_ids[0]
    artifact_fingerprint = _canonical_hash([item[2] for item in selections])
    export_recipe_hash = _canonical_hash(
        {
            "previews": [item[1]["recipe_hash"] for item in selections],
            "recompute_statistics": recompute_statistics,
        }
    )

    dataset_id = manifest["source"]["dataset_id"]
    source_root, _ = load_source(
        database, settings, dataset_id, manifest["fingerprint"]
    )
    _report_progress(
        progress,
        stage="preparing",
        completed=1,
        total=1,
        unit="items",
        current_item="승인된 미리보기 확인",
    )
    derived = settings.nas_root / "derived"
    if derived.is_symlink() or not derived.is_dir():
        raise SegmentationError("Derived dataset storage is unavailable")
    final = derived / output_name
    provenance_path = settings.nas_root / "manifests/segmentation" / f"{job_id}.json"
    with _output_lock(settings, output_name):
        database.assert_job_lease(job_id, worker_id=worker_id)
        if final.exists() or final.is_symlink():
            result = _reuse_export(
                database,
                settings,
                final=final,
                provenance_path=provenance_path,
                job_id=job_id,
                preview_id=str(preview_id),
                output_name=output_name,
                recipe_hash=export_recipe_hash,
                artifact_fingerprint=artifact_fingerprint,
            )
            _report_progress(
                progress,
                stage="complete",
                completed=1,
                total=1,
                unit="items",
                current_item="기존 Segmentation 결과 재사용",
                force=True,
            )
            return result
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".incoming-segmentation-{job_id}-{uuid.uuid4().hex}-",
                dir=derived,
            )
        )
        destination = staging / output_name
        try:
            database.assert_job_lease(job_id, worker_id=worker_id)
            _report_progress(
                progress,
                stage="publish",
                completed=0,
                total=0,
                unit="bytes",
                current_item="원본 데이터셋 복사",
                force=True,
            )
            shutil.copytree(source_root, destination)
            copied_fingerprint = dataset_content_fingerprint(destination)
            if copied_fingerprint != manifest["fingerprint"]:
                raise SegmentationError("Source dataset copy verification failed")
            if dataset_content_fingerprint(source_root, reuse_file_digests=True) != manifest["fingerprint"]:
                raise RecipeRevisionMismatchError(dataset_id)

            def lease() -> None:
                database.assert_job_lease(job_id, worker_id=worker_id)

            from datasetui.segmentation_export import (
                POLICY as VIDEO_POLICY,
                write_segmented_videos,
            )

            # Untouched episodes keep their exact source bytes; each approved
            # episode/camera gets a new source-codec file and repointed metadata.
            written_videos = write_segmented_videos(
                destination,
                [(item_root, item_manifest) for item_root, item_manifest, _ in selections],
                lease,
                on_progress=progress,
            )
            _report_progress(
                progress,
                stage="statistics",
                completed=0,
                total=0,
                unit="frames",
                current_item="영상 통계 재계산",
                force=True,
            )
            from datasetui.deferred_statistics import (
                preserve_deferred_statistics,
                read_deferred_statistics,
            )

            # Relative artifacts embed normalization data and cannot be deferred.
            relative_required = (destination / "meta/relative_action.json").exists()
            if recompute_statistics or relative_required:
                if read_deferred_statistics(source_root):
                    from datasetui.output_statistics import write_output_statistics

                    write_output_statistics(destination)
                else:
                    for _, item_manifest, _ in selections:
                        _update_image_statistics(
                            destination,
                            item_manifest["spec"]["video_key"],
                            int(item_manifest["spec"]["episode_index"]),
                        )
                statistics = {"status": "recomputed"}
            else:
                statistics = preserve_deferred_statistics(source_root, destination)
            _report_progress(
                progress,
                stage="statistics",
                completed=1,
                total=1,
                unit="items",
                current_item="영상 통계 재계산"
                if statistics["status"] == "recomputed"
                else "분포 통계 미재계산 표시",
                force=True,
            )
            _report_progress(
                progress,
                stage="validate",
                completed=0,
                total=2,
                unit="items",
                current_item="전체 검사",
                force=True,
            )
            full_gate = validate_dataset_root(destination, mode="full")
            _report_progress(
                progress,
                stage="validate",
                completed=1,
                total=2,
                unit="items",
                current_item="전체 검사",
            )
            export_gate = validate_dataset_root(destination, mode="export_gate")
            if not full_gate["passed"] or not export_gate["passed"]:
                raise SegmentationError("Segmented dataset failed the export gate")
            _report_progress(
                progress,
                stage="validate",
                completed=2,
                total=2,
                unit="items",
                current_item="Export Gate 검사",
            )
            for item_root, _, item_fingerprint in selections:
                if dataset_content_fingerprint(item_root, reuse_file_digests=True) != item_fingerprint:
                    raise SegmentationError(
                        "Approved preview artifacts changed during export"
                    )
            if dataset_content_fingerprint(source_root, reuse_file_digests=True) != manifest["fingerprint"]:
                raise RecipeRevisionMismatchError(dataset_id)
            output_manifest = dataset_content_manifest(destination)
            provenance = {
                "schema_version": 1,
                "job_id": job_id,
                "preview_id": str(preview_id),
                "preview_ids": selected_ids,
                "recipe_hash": export_recipe_hash,
                "source_dataset_id": dataset_id,
                "source_fingerprint": manifest["fingerprint"],
                "artifact_fingerprint": artifact_fingerprint,
                "output_name": output_name,
                "output_manifest": output_manifest,
                "model": manifest["model"],
                "statistics": statistics,
                "video_policy": VIDEO_POLICY,
                "segmented_videos": written_videos,
            }
            _write_json_atomic(provenance_path, provenance)
            database.assert_job_lease(job_id, worker_id=worker_id)
            if final.exists() or final.is_symlink():
                raise SegmentationError("Segmentation output name already exists")
            _report_progress(
                progress,
                stage="publish",
                completed=0,
                total=1,
                unit="items",
                current_item="Segmentation 데이터셋 게시",
                force=True,
            )
            database.begin_job_finalization(job_id, worker_id=worker_id)
            destination.rename(final)
            _report_progress(
                progress,
                stage="publish",
                completed=1,
                total=1,
                unit="items",
                current_item="Segmentation 데이터셋 게시",
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    _report_progress(
        progress,
        stage="register",
        completed=0,
        total=1,
        unit="items",
        current_item="라이브러리 갱신",
        force=True,
    )
    result = _register_export(database, settings, output_name, output_manifest)
    _report_progress(
        progress,
        stage="register",
        completed=1,
        total=1,
        unit="items",
        current_item="라이브러리 갱신",
    )
    _report_progress(
        progress,
        stage="complete",
        completed=1,
        total=1,
        unit="items",
        current_item="Segmentation 데이터셋 완료",
        force=True,
    )
    return result


def _estimated_sam_passes(parsed: SegmentationSpec) -> int:
    from datasetui.sam3_engine import _expected_passes

    groups: dict[Any, list[Any]] = {}
    for index, prompt in enumerate(parsed.prompts):
        groups.setdefault(prompt.object_id or f"p{index}", []).append(prompt)
    for correction in parsed.corrections:
        if correction.object_id in groups:
            groups[correction.object_id].append(groups[correction.object_id][0])
    return _expected_passes(groups) if groups else 1


def _engine_progress(engine: Any, progress: Any) -> dict[str, Any]:
    """Forward SAM frame progress when the engine supports it (fixtures may not)."""
    import inspect

    try:
        accepts = "on_progress" in inspect.signature(engine.propagate).parameters
    except (TypeError, ValueError):
        accepts = False
    if not accepts or progress is None:
        return {}

    return {"on_progress": progress}


def _default_engine(settings: Settings, *, mixed_spatial: bool = False) -> Any:
    from datasetui.sam3_engine import Sam3Engine

    return Sam3Engine(settings, mixed_spatial=mixed_spatial)


def _episode(
    source: _DatasetSource, episode_index: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    total = int(source.info.get("total_episodes", 0))
    if episode_index < 0 or episode_index >= total:
        raise SegmentationError("Episode is outside the dataset")
    return source.episode(episode_index)


def _validate_video_selection(
    source: _DatasetSource, video_key: str, frame_index: int, frame_count: int
) -> None:
    if video_key not in source.video_keys:
        raise SegmentationError("Video key is not available")
    if frame_index < 0 or frame_index >= frame_count:
        raise SegmentationError("Frame is outside the episode")


def _safe_regular_path(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SegmentationError("Dataset file path is unsafe") from exc
    current = root
    for index, component in enumerate(relative.parts):
        if component in {"", ".", ".."}:
            raise SegmentationError("Dataset file path is unsafe")
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SegmentationError("Dataset file is unavailable") from exc
        if current.is_symlink():
            raise SegmentationError("Dataset file path is unsafe")
        final = index == len(relative.parts) - 1
        if final and not stat.S_ISREG(metadata.st_mode):
            raise SegmentationError("Dataset file is unsafe")
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise SegmentationError("Dataset file path is unsafe")
    return current


def _safe_directory_path(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SegmentationError("Dataset directory path is unsafe") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SegmentationError("Dataset directory is unavailable") from exc
        if current.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise SegmentationError("Dataset directory path is unsafe")
    return current


def _write_episode_clip(
    source_path: Path,
    start: int,
    count: int,
    fps: float,
    output_path: Path,
    check_lease: Callable[[], None],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    current_item: str | None = None,
) -> tuple[int, int]:
    frames = _iter_video_arrays(source_path)
    output = None
    stream = None
    written = 0
    width = height = 0
    _report_progress(
        on_progress,
        stage="read",
        completed=0,
        total=count,
        unit="frames",
        current_item=current_item,
        force=True,
    )
    try:
        for absolute_index, array in enumerate(frames):
            if absolute_index < start:
                continue
            if absolute_index >= start + count:
                break
            if output is None:
                height, width = array.shape[:2]
                if width * height > MAX_IMAGE_PIXELS:
                    raise SegmentationError("Video frame exceeds the image pixel limit")
                output, stream = _open_video_writer(output_path, fps, width, height)
            _encode_array(output, stream, array)
            written += 1
            _report_progress(
                on_progress,
                stage="read",
                completed=written,
                total=count,
                unit="frames",
                current_item=current_item,
            )
            if written % 32 == 0:
                check_lease()
    finally:
        if output is not None and stream is not None:
            _close_video_writer(output, stream)
    if written != count:
        raise SegmentationError("Source video does not cover the full episode")
    return width, height


def _render_preview(
    root: Path,
    mode: str,
    background: Image.Image,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    check_lease: Callable[[], None],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    has_keep: bool = True,
) -> None:
    composite, composite_stream = _open_video_writer(
        root / "composite.mp4", fps, width, height
    )
    mask_video, mask_stream = _open_video_writer(root / "mask.mp4", fps, width, height)
    background_array = np.asarray(background, dtype=np.uint8)
    seen = 0
    _report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=frame_count,
        unit="frames",
        current_item="미리보기 영상 생성",
        force=True,
    )
    try:
        for frame_index, original in enumerate(
            _iter_video_arrays(root / "original.mp4")
        ):
            if frame_index >= frame_count:
                break
            replace = _read_mask(root, "replace", frame_index, width, height)
            protect = _read_mask(root, "protect", frame_index, width, height)
            selected = ~retained_mask(mode, protect, replace, has_keep=has_keep)
            output = np.where(selected[:, :, None], background_array, original)
            mask_rgb = np.repeat(
                (selected.astype(np.uint8) * 255)[:, :, None], 3, axis=2
            )
            _encode_array(composite, composite_stream, output)
            _encode_array(mask_video, mask_stream, mask_rgb)
            seen += 1
            _report_progress(
                on_progress,
                stage="write",
                completed=seen,
                total=frame_count,
                unit="frames",
                current_item="미리보기 영상 생성",
            )
            if seen % 32 == 0:
                check_lease()
    finally:
        _close_video_writer(composite, composite_stream)
        _close_video_writer(mask_video, mask_stream)
    if seen != frame_count:
        raise SegmentationError("Preview rendering lost source frames")


def _apply_corrections(
    root: Path,
    spec: SegmentationSpec,
    count: int,
    width: int,
    height: int,
    *,
    include_object_corrections: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    if spec.mode == "object_selection":
        return
    by_frame: dict[tuple[str, int], list[Any]] = {}
    for correction in spec.corrections:
        if correction.object_id is not None and not include_object_corrections:
            continue
        by_frame.setdefault((correction.target, correction.frame_index), []).append(
            correction
        )
    total = count * len(MASK_TARGETS)
    completed = 0
    _report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=total,
        unit="items",
        current_item="마스크 보정 반영",
        force=True,
    )
    for target in MASK_TARGETS:
        directory = root / target
        if directory.is_symlink() or not directory.is_dir():
            raise SegmentationError("Segmentation engine omitted a mask directory")
        for frame_index in range(count):
            path = directory / f"{frame_index:06d}.png"
            mask = _mask_image(path, width, height)
            draw = ImageDraw.Draw(mask)
            for correction in by_frame.get((target, frame_index), []):
                radius = correction.radius * width
                fill = 255 if correction.operation == "add" else 0
                for point in correction.points:
                    x = point.x * (width - 1)
                    y = point.y * (height - 1)
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius), fill=fill
                    )
            mask.save(path, format="PNG", compress_level=6)
            completed += 1
            _report_progress(
                on_progress,
                stage="write",
                completed=completed,
                total=total,
                unit="items",
                current_item=f"{target} 마스크 · 프레임 {frame_index}",
            )


def _read_mask(
    root: Path, target: str, index: int, width: int, height: int
) -> np.ndarray:
    return (
        np.asarray(
            _mask_image(root / target / f"{index:06d}.png", width, height),
            dtype=np.uint8,
        )
        >= 128
    )


def _mask_image(path: Path, width: int, height: int) -> Image.Image:
    if path.is_symlink() or not path.is_file():
        raise SegmentationError("Segmentation engine omitted a frame mask")
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise SegmentationError(
                    "Segmentation mask dimensions do not match video"
                )
            binary = np.where(np.asarray(image.convert("L")) >= 128, 255, 0).astype(
                np.uint8
            )
            return Image.fromarray(binary, mode="L")
    except OSError as exc:
        raise SegmentationError("Segmentation mask could not be read") from exc




def _update_image_statistics(root: Path, video_key: str, episode_index: int) -> None:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    source = _DatasetSource(root, info)
    paths: dict[Path, list[tuple[int, int]]] = {}
    for index in range(int(info["total_episodes"])):
        data, metadata = source.episode(index)
        path, start = source.video_source(index, video_key, metadata)
        paths.setdefault(path, []).append((start, len(data)))
    episode_image_stats = [
        _image_stats([path], ranges={path: [(start, length)]})
        for path, selections in paths.items()
        for start, length in selections
    ]
    global_stats = _aggregate_image_stats(episode_image_stats)
    stats_path = root / "meta/stats.json"
    stats = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.is_file()
        else {}
    )
    if not isinstance(stats, dict):
        raise SegmentationError("Dataset statistics are invalid")
    stats[video_key] = global_stats
    _write_json_atomic(stats_path, stats)

    episode_stats_path = root / "meta/episodes_stats.jsonl"
    if episode_stats_path.is_file():
        rows = [
            json.loads(line)
            for line in episode_stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        data, metadata = source.episode(episode_index)
        video_path, start = source.video_source(episode_index, video_key, metadata)
        selected_stats = _image_stats(
            [video_path], ranges={video_path: [(start, len(data))]}
        )
        matched = False
        for row in rows:
            if int(row.get("episode_index", -1)) == episode_index:
                row.setdefault("stats", {})[video_key] = selected_stats
                matched = True
        if not matched:
            raise SegmentationError("Per-episode statistics are incomplete")
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        episode_stats_path.write_text(text, encoding="utf-8")
    if str(info.get("codebase_version")) == "v3.0":
        data, metadata = source.episode(episode_index)
        video_path, start = source.video_source(episode_index, video_key, metadata)
        selected_stats = _image_stats(
            [video_path], ranges={video_path: [(start, len(data))]}
        )
        _update_v3_flattened_stats(root, episode_index, video_key, selected_stats)
    episode_stats_root = root / "meta/episodes_stats"
    if episode_stats_root.is_dir() and not episode_stats_root.is_symlink():
        data, metadata = source.episode(episode_index)
        video_path, start = source.video_source(episode_index, video_key, metadata)
        selected_stats = _image_stats(
            [video_path], ranges={video_path: [(start, len(data))]}
        )
        matched = False
        for parquet_path in sorted(episode_stats_root.rglob("*.parquet")):
            if parquet_path.is_symlink():
                raise SegmentationError("Per-episode statistics path is unsafe")
            frame = pd.read_parquet(parquet_path)
            if "episode_index" not in frame.columns:
                raise SegmentationError("Per-episode statistics are invalid")
            rows = frame.index[frame["episode_index"] == episode_index].tolist()
            for row_index in rows:
                if "stats" in frame.columns and isinstance(
                    frame.at[row_index, "stats"], dict
                ):
                    value = dict(frame.at[row_index, "stats"])
                    value[video_key] = selected_stats
                    frame.at[row_index, "stats"] = value
                elif video_key in frame.columns:
                    frame.at[row_index, video_key] = selected_stats
                else:
                    raise SegmentationError(
                        "Per-episode image statistics schema is unsupported"
                    )
                matched = True
            if rows:
                temporary = parquet_path.with_name(f".{parquet_path.name}.tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(parquet_path)
        if not matched:
            raise SegmentationError("Per-episode statistics are incomplete")


def _image_stats(
    paths: Any, *, ranges: dict[Path, list[tuple[int, int]]] | None = None
) -> dict[str, Any]:
    minimum = np.full(3, 255, dtype=np.uint8)
    maximum = np.zeros(3, dtype=np.uint8)
    sums = np.zeros(3, dtype=np.float64)
    sums_squared = np.zeros(3, dtype=np.float64)
    histograms = np.zeros((3, 256), dtype=np.int64)
    pixel_count = 0
    frame_count = 0
    for path in paths:
        for index, frame in enumerate(_iter_video_arrays(path)):
            selected_ranges = (ranges or {}).get(path)
            if selected_ranges is not None and not any(
                start <= index < start + length for start, length in selected_ranges
            ):
                continue
            flat = frame.reshape(-1, 3)
            minimum = np.minimum(minimum, flat.min(axis=0))
            maximum = np.maximum(maximum, flat.max(axis=0))
            sums += flat.sum(axis=0)
            sums_squared += np.square(flat.astype(np.float64)).sum(axis=0)
            for channel in range(3):
                histograms[channel] += np.bincount(flat[:, channel], minlength=256)
            pixel_count += len(flat)
            frame_count += 1
    if frame_count == 0:
        raise SegmentationError("Video statistics could not be computed")
    mean = sums / pixel_count
    std = np.sqrt(np.maximum(0, sums_squared / pixel_count - np.square(mean)))
    result: dict[str, Any] = {
        "min": (minimum.astype(float) / 255.0).reshape(3, 1, 1).tolist(),
        "max": (maximum.astype(float) / 255.0).reshape(3, 1, 1).tolist(),
        "mean": (mean / 255.0).reshape(3, 1, 1).tolist(),
        "std": (std / 255.0).reshape(3, 1, 1).tolist(),
        "count": [frame_count],
    }
    for name, quantile in (
        ("q01", 0.01),
        ("q10", 0.1),
        ("q50", 0.5),
        ("q90", 0.9),
        ("q99", 0.99),
    ):
        threshold = max(0, int(np.ceil(quantile * pixel_count)) - 1)
        values = [
            int(np.searchsorted(np.cumsum(histograms[channel]), threshold + 1))
            for channel in range(3)
        ]
        result[name] = (
            (np.asarray(values, dtype=float) / 255.0).reshape(3, 1, 1).tolist()
        )
    return result


def _aggregate_image_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise SegmentationError("Video statistics could not be aggregated")
    counts = np.asarray([item["count"][0] for item in items], dtype=np.float64)
    total = float(counts.sum())
    means = np.asarray([item["mean"] for item in items], dtype=np.float64)
    deviations = np.asarray([item["std"] for item in items], dtype=np.float64)
    weights = counts.reshape((-1, 1, 1, 1))
    mean = np.sum(means * weights, axis=0) / total
    variance = (
        np.sum((np.square(deviations) + np.square(means - mean)) * weights, axis=0)
        / total
    )
    aggregated: dict[str, Any] = {
        "min": np.min(np.asarray([item["min"] for item in items]), axis=0).tolist(),
        "max": np.max(np.asarray([item["max"] for item in items]), axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [int(total)],
    }
    for name in ("q01", "q10", "q50", "q90", "q99"):
        values = np.asarray([item[name] for item in items], dtype=np.float64)
        aggregated[name] = (np.sum(values * weights, axis=0) / total).tolist()
    return aggregated


def _update_v3_flattened_stats(
    root: Path, episode_index: int, video_key: str, stats: dict[str, Any]
) -> None:
    metadata_root = root / "meta/episodes"
    matched = False
    stats_present = False
    for path in sorted(metadata_root.rglob("*.parquet")):
        if path.is_symlink():
            raise SegmentationError("Episode metadata path is unsafe")
        frame = pd.read_parquet(path)
        stat_columns = [name for name in frame.columns if name.startswith("stats/")]
        stats_present = stats_present or bool(stat_columns)
        if "episode_index" not in frame.columns:
            raise SegmentationError("Episode metadata is invalid")
        rows = frame.index[frame["episode_index"] == episode_index].tolist()
        if not rows:
            continue
        required = {f"stats/{video_key}/{name}" for name in stats}
        if stat_columns and not required.issubset(frame.columns):
            raise SegmentationError(
                "Flattened v3 image statistics schema is incomplete"
            )
        if not stat_columns:
            continue
        for row_index in rows:
            for stat_name, value in stats.items():
                frame.at[row_index, f"stats/{video_key}/{stat_name}"] = value
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        matched = True
    if stats_present and not matched:
        raise SegmentationError("Flattened v3 image statistics are incomplete")


def _iter_video_arrays(path: Path):
    if path.is_symlink() or not path.is_file():
        raise SegmentationError("Video file is unavailable")
    import av

    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def _decode_one(path: Path, index: int) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise SegmentationError("Video file is unavailable")
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.base_rate or 0)
        time_base = float(stream.time_base)
        if rate <= 0 or time_base <= 0:
            raise SegmentationError("Video timing metadata is unavailable")
        start_pts = int(stream.start_time or 0)
        target_pts = start_pts + int((index / rate) / time_base)
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_index = round((int(frame.pts) - start_pts) * time_base * rate)
            if frame_index == index:
                return frame.to_ndarray(format="rgb24")
            if frame_index > index:
                break
    finally:
        container.close()
    raise SegmentationError("Video frame is unavailable")


def _video_frame_count(path: Path) -> int:
    return sum(1 for _ in _iter_video_arrays(path))


def _open_video_writer(path: Path, fps: float, width: int, height: int):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(str(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return container, stream






def _encode_array(container: Any, stream: Any, array: np.ndarray) -> None:
    import av

    frame = av.VideoFrame.from_ndarray(
        np.asarray(array, dtype=np.uint8), format="rgb24"
    )
    for packet in stream.encode(frame):
        container.mux(packet)


def _close_video_writer(container: Any, stream: Any) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _output_lock(settings: Settings, output_name: str):
    lock_root = settings.nas_root / "manifests/segmentation/.locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(output_name.encode()).hexdigest() + ".lock"
    with (lock_root / lock_name).open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reuse_export(
    database: Database,
    settings: Settings,
    *,
    final: Path,
    provenance_path: Path,
    job_id: str,
    preview_id: str,
    output_name: str,
    recipe_hash: str,
    artifact_fingerprint: str,
) -> dict[str, Any]:
    if final.is_symlink() or not final.is_dir() or not provenance_path.is_file():
        raise SegmentationError("Segmentation output name already exists")
    provenance = json.loads(
        _read_regular_bytes(provenance_path, max_bytes=MAX_MANIFEST_BYTES)
    )
    if (
        provenance.get("job_id") != job_id
        or provenance.get("preview_id") != preview_id
        or provenance.get("output_name") != output_name
        or provenance.get("recipe_hash") != recipe_hash
        or provenance.get("artifact_fingerprint") != artifact_fingerprint
    ):
        raise SegmentationError("Segmentation output name already exists")
    output_manifest = dataset_content_manifest(final)
    if output_manifest != provenance.get("output_manifest"):
        raise SegmentationError("Published segmentation output was modified")
    return _register_export(database, settings, output_name, output_manifest)


def _register_export(
    database: Database,
    settings: Settings,
    output_name: str,
    output_manifest: dict[str, Any],
) -> dict[str, Any]:
    derived = settings.nas_root / "derived"
    generation = database.begin_dataset_scan("derived")
    database.synchronize_datasets(
        storage_area="derived",
        records=scan_storage_area(
            settings.nas_root,
            "derived",
            max_depth=settings.dataset_scan_max_depth,
        ),
        scan_generation=generation,
    )
    candidate = inspect_dataset(
        area_root=derived, storage_area="derived", relative_path=output_name
    )
    dataset = database.get_dataset(
        next(
            item["id"]
            for item in database.list_datasets(storage_area="derived")
            if item["relative_path"] == output_name
            and item["fingerprint"] == candidate.fingerprint
        )
    )
    return {
        "dataset_id": dataset["id"],
        "output_name": output_name,
        "relative_path": output_name,
        "fingerprint": dataset["fingerprint"],
        "manifest_sha256": output_manifest["tree_sha256"],
    }


def _read_manifest(root: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular_bytes(root / "manifest.json", max_bytes=MAX_MANIFEST_BYTES)
        )
    except Exception as exc:
        raise SegmentationError("Preview manifest is unavailable") from exc
    if not isinstance(value, dict):
        raise SegmentationError("Preview manifest is invalid")
    return value


def _preview_artifact_manifest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    total = 0
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise SegmentationError("Preview artifact directory is unsafe")
        files.extend(current_path / name for name in names if name != "manifest.json")
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise SegmentationError("Preview artifact is unsafe")
        file_hash = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_hash.update(chunk)
                size += len(chunk)
        relative = path.relative_to(root).as_posix()
        hashes[relative] = file_hash.hexdigest()
        digest.update(f"{relative}\0{size}\0{file_hash.hexdigest()}\n".encode())
        total += size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
        "file_sha256": hashes,
    }


def _reuse_preview(
    root: Path, recipe_hash: str, fingerprint: str, preview_id: str
) -> dict[str, Any]:
    manifest = _read_manifest(root)
    if (
        manifest.get("preview_id") != preview_id
        or manifest.get("recipe_hash") != recipe_hash
        or manifest.get("fingerprint") != fingerprint
        or manifest.get("artifacts") != _preview_artifact_manifest(root)
    ):
        raise SegmentationError("Preview output conflicts with an earlier attempt")
    return _preview_result(manifest, dataset_content_fingerprint(root, reuse_file_digests=True), reused=True)


def _preview_result(
    manifest: dict[str, Any], artifact_fingerprint: str, *, reused: bool = False
) -> dict[str, Any]:
    provenance = manifest["model"]
    model = str(provenance.get("model") or provenance.get("engine") or "SAM 3.1")
    file_hashes = manifest["artifacts"]["file_sha256"]
    return {
        "preview_id": manifest["preview_id"],
        "recipe_hash": manifest["recipe_hash"],
        "fingerprint": manifest["fingerprint"],
        "frame_count": manifest["frame_count"],
        "model": model,
        "model_provenance": provenance,
        "mask_input_hash": manifest.get("mask_input_hash"),
        "candidates": provenance.get("candidates", []),
        "selection_required": manifest.get("selection_required", False),
        "review_signals": manifest.get("review_signals", []),
        "review_warnings": manifest.get("review_signals", []),
        "review_blocked": manifest.get("review_blocked", False),
        "object_coverage": manifest.get("object_coverage", []),
        "render_mode": manifest["spec"].get("render_mode", "image"),
        "artifact_fingerprint": artifact_fingerprint,
        "artifact_hashes": {
            name: file_hashes[name]
            for name in file_hashes
            if name in ("original.mp4", "composite.mp4", "mask.mp4")
            or name.startswith("candidate-")
        },
        "reused": reused,
    }
