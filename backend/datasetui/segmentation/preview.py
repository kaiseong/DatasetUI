"""Episode previews: creation, mask reuse, verification and approval checks."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from PIL import Image

from datasetui.config import Settings
from datasetui.content_integrity import (
    dataset_content_fingerprint,
)
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.job_progress import JobProgressReporter
from datasetui.segmentation import engine as engine_module
from datasetui.segmentation.contract import (
    PendingPreviewSpec,
    SegmentationSpec,
    decode_background,
)
from datasetui.segmentation.engine import engine_progress, estimated_sam_passes
from datasetui.segmentation.errors import SegmentationError
from datasetui.segmentation.media import (
    MAX_IMAGE_PIXELS,
    close_video_writer,
    decode_frame,
    encode_frame,
    iter_video_arrays,
    open_video_writer,
    png_bytes,
    video_frame_count,
    write_episode_clip,
)
from datasetui.segmentation.paths import output_lock, safe_regular_path
from datasetui.segmentation.selection import (
    MASK_TARGETS,
    apply_legacy_corrections,
    brush_hints,
    has_keep_objects,
    mask_inputs,
    read_mask,
    retained_mask,
    validate_detections,
)
from datasetui.segmentation.source import (
    _episode,
    load_source,
    validate_video_selection,
)
from datasetui.transforms import (
    _read_regular_bytes,
    _report_progress,
    _write_json_atomic,
)

MAX_MANIFEST_BYTES = 16 * 1024 * 1024


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

    spec = resolve_preview_spec(database, settings, spec)
    parsed = SegmentationSpec.model_validate(spec)
    from datasetui.segmentation.candidates import validate_candidate_references
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
    validate_video_selection(source, parsed.video_key, 0, frame_count)
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
            else frame_count * 4 * estimated_sam_passes(parsed),
            "write": frame_count * 1.5,
            "validate": frame_count * 0.3,
            "publish": frame_count * 0.1,
        },
    )

    source_path, source_start = source.video_source(
        parsed.episode_index, parsed.video_key, metadata
    )
    source_path = safe_regular_path(root, source_path)
    probe = decode_frame(source_path, source_start)
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
    background_bytes = png_bytes(background)
    background_sha256 = hashlib.sha256(background_bytes).hexdigest()
    stored_spec = {**normalized, "background_sha256": background_sha256}
    stored_spec.pop("background_base64", None)
    recipe_hash = canonical_hash(stored_spec)
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
        from datasetui.segmentation.selection import (
            apply_object_corrections,
            apply_protected_regions,
            apply_selection,
            mask_inputs,
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
            clip_width, clip_height = write_episode_clip(
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
        segmenter = engine or engine_module.default_engine(
            settings, mixed_spatial=parsed.mode == "object_selection"
        )
        provenance = cached_provenance or segmenter.propagate(
            video_path=staging / "original.mp4",
            prompts=prompt_payload,
            frame_count=frame_count,
            output_dir=staging,
            check_lease=lease,
            **engine_progress(segmenter, progress),
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
        apply_legacy_corrections(
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
        from datasetui.segmentation.selection import (
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
                "total_frames": video_frame_count(source_path),
            },
            "model": provenance,
            "mask_input_hash": canonical_hash(mask_inputs(normalized)),
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
        with output_lock(settings, f"preview-{job_id}"):
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
    composite, composite_stream = open_video_writer(
        root / "composite.mp4", fps, width, height
    )
    mask_video, mask_stream = open_video_writer(root / "mask.mp4", fps, width, height)
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
            iter_video_arrays(root / "original.mp4")
        ):
            if frame_index >= frame_count:
                break
            replace = read_mask(root, "replace", frame_index, width, height)
            protect = read_mask(root, "protect", frame_index, width, height)
            selected = ~retained_mask(mode, protect, replace, has_keep=has_keep)
            output = np.where(selected[:, :, None], background_array, original)
            mask_rgb = np.repeat(
                (selected.astype(np.uint8) * 255)[:, :, None], 3, axis=2
            )
            encode_frame(composite, composite_stream, output)
            encode_frame(mask_video, mask_stream, mask_rgb)
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
        close_video_writer(composite, composite_stream)
        close_video_writer(mask_video, mask_stream)
    if seen != frame_count:
        raise SegmentationError("Preview rendering lost source frames")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_manifest(root: Path) -> dict[str, Any]:
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
    manifest = read_manifest(root)
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


def preview_directory(settings: Settings, preview_id: str) -> Path:
    preview_id = str(UUID(preview_id))
    root = settings.jobs_root / "segmentation"
    path = root / preview_id
    if root.is_symlink() or path.is_symlink():
        raise ValueError("미리보기 경로가 올바르지 않습니다.")
    return path


def verified_preview(
    database: Database,
    settings: Settings,
    preview_id: str,
    *,
    verify_source: bool = True,
):
    job = database.get_job(preview_id)
    if job["kind"] != "segmentation.preview" or job["status"] != "succeeded":
        raise ValueError("완료된 미리보기가 필요합니다.")
    result = job["result"]
    path = preview_directory(settings, preview_id)
    if not result or result.get("artifact_fingerprint") != dataset_content_fingerprint(
        path, reuse_file_digests=True
    ):
        raise ValueError("미리보기 파일이 변경되었습니다. 다시 생성하세요.")
    provenance = result.get("model_provenance") or {}
    if (
        provenance.get("engine") == "sam3.1-multiplex"
        and "candidates" not in provenance
    ):
        raise ValueError(
            "객체 선택 이전 버전의 미리보기입니다. 새 미리보기를 생성하세요."
        )
    from datasetui.segmentation.source import load_source

    if verify_source:
        load_source(
            database,
            settings,
            job["payload"]["spec"]["dataset_id"],
            result["fingerprint"],
        )
    return effective_preview_job(job), result


def verify_approval(
    database: Database,
    settings: Settings,
    *,
    preview_id: str,
    profile_id: str,
    approval_token: str,
    verify_source: bool = True,
):
    job, result = verified_preview(
        database, settings, preview_id, verify_source=verify_source
    )
    if result.get("selection_required"):
        raise ValueError("객체 후보 선택이 완료되지 않았습니다.")
    if result.get("review_blocked"):
        raise ValueError(
            "영상 전체에서 대상 영역이 비어 있습니다. 프롬프트를 보정하세요."
        )
    if job["profile_id"] != profile_id:
        raise ValueError("미리보기를 만든 프로필로 승인하세요.")
    with database.connect() as connection:
        approval = connection.execute(
            "SELECT * FROM segmentation_approvals WHERE preview_id = ? AND profile_id = ?",
            (preview_id, profile_id),
        ).fetchone()
    token_hash = hashlib.sha256(approval_token.encode()).hexdigest()
    if (
        approval is None
        or not secrets.compare_digest(approval["token_hash"], token_hash)
        or approval["artifact_fingerprint"] != result["artifact_fingerprint"]
        or approval["recipe_hash"] != result["recipe_hash"]
    ):
        raise ValueError("현재 미리보기를 승인한 뒤 다시 시도하세요.")
    return job, result


def reuse_masks(
    database, settings, parsed, staging: Path, *, owner_profile_id: str
) -> dict:
    from datasetui.segmentation.errors import SegmentationError
    from datasetui.segmentation.paths import safe_regular_path

    job, result = verified_preview(database, settings, str(parsed.source_preview_id))
    if job["profile_id"] != owner_profile_id:
        raise SegmentationError("Mask reuse requires the original preview profile")
    previous = SegmentationSpec.model_validate(job["payload"]["spec"])
    if mask_inputs(previous.model_dump(mode="json")) != mask_inputs(
        parsed.model_dump(mode="json")
    ):
        raise SegmentationError(
            "Mask inputs changed; generate a new segmentation preview"
        )
    root = settings.jobs_root / "segmentation" / str(parsed.source_preview_id)
    manifest = read_manifest(root)
    if parsed.mode == "object_selection":
        from datasetui.segmentation.engine import MIXED_HINT_POLICY

        if manifest.get("model", {}).get("hint_policy") != MIXED_HINT_POLICY:
            raise SegmentationError(
                "혼합 힌트 처리 방식이 변경되었습니다. 기존 마스크 재사용 대신 새로 분할하세요."
            )
    for name in (
        "original.mp4",
        *[item["artifact_name"] for item in result.get("candidates", [])],
    ):
        shutil.copyfile(safe_regular_path(root, root / name), staging / name)
    for name in ("replace", "protect", "instances"):
        source = root / name
        if source.is_dir():
            # verified_preview fingerprint rejects symlinks before this copy.
            shutil.copytree(source, staging / name)
    return {**manifest["model"], "mask_cache_source": str(parsed.source_preview_id)}


def resolve_preview_spec(database, settings, spec):
    if spec.get("fingerprint"):
        return spec
    from datasetui.segmentation.frames import read_snapshot_frame
    from datasetui.segmentation.source import load_source

    parsed = PendingPreviewSpec.model_validate(spec)

    def verify():
        read_snapshot_frame(
            database,
            settings,
            str(parsed.dataset_id),
            str(parsed.frame_token),
            parsed.episode_index,
            parsed.video_key,
            0,
            verify_only=True,
        )

    verify()
    _, source = load_source(database, settings, str(parsed.dataset_id))
    verify()
    result = parsed.model_dump(mode="json", exclude={"frame_token"})
    result["fingerprint"] = source.segmentation_fingerprint
    return result


def effective_preview_job(job):
    """Expose the validated execution identity without mutating idempotent input."""
    spec = job["payload"]["spec"]
    if spec.get("fingerprint") or not job.get("result"):
        return job
    spec = {key: value for key, value in spec.items() if key != "frame_token"}
    spec["fingerprint"] = job["result"]["fingerprint"]
    return {**job, "payload": {**job["payload"], "spec": spec}}
