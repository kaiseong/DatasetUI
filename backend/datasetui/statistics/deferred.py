"""Explicitly inherited statistics for outputs normalized later by training."""

from __future__ import annotations

import json
from pathlib import Path

from datasetui.content_integrity import hash_regular_file
from datasetui.transform_errors import CurationTransformError


POLICY = "datasetui-distribution-statistics-deferred-v1"


INFO_KEY = "datasetui_statistics_status"


MARKER = "datasetui_statistics.json"


NOTICE_START = "<!-- datasetui-deferred-statistics -->"


NOTICE_END = "<!-- /datasetui-deferred-statistics -->"


def should_defer(trim_config, relative_action, *, inherited=False):
    # Relative output includes its own normalization artifact, which requires
    # verified absolute and relative distributions.
    if (relative_action or {}).get("enabled", False):
        return False
    if trim_config.get("enabled", False):
        return not trim_config.get("recompute_statistics", True)
    return inherited


def _metadata_hashes(root: Path) -> dict[str, str]:
    metadata = root / "meta"
    if metadata.is_symlink() or not metadata.is_dir():
        raise CurationTransformError("Deferred statistics metadata is unsafe")
    paths = [metadata / "info.json", metadata / "stats.json"]
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        path = metadata / name
        if path.exists() or path.is_symlink():
            paths.append(path)
    episodes = metadata / "episodes"
    if episodes.is_symlink():
        raise CurationTransformError("Deferred episode metadata is unsafe")
    if episodes.is_dir():
        for path in sorted(episodes.rglob("*")):
            if path.is_symlink():
                raise CurationTransformError("Deferred episode metadata is unsafe")
            if not path.is_dir():
                paths.append(path)
    return {str(path.relative_to(root)): hash_regular_file(path)[1] for path in paths}


def read_deferred_statistics(root: Path) -> bool:
    from datasetui.dataset_io.files import read_json

    info = read_json(root / "meta/info.json")
    marker = root / "meta" / MARKER
    declared = info.get(INFO_KEY)
    if declared is None and not marker.exists() and not marker.is_symlink():
        return False
    try:
        value = read_json(marker)
        if (root / "meta/relative_action.json").exists() or (
            root / "meta/relative_action.json"
        ).is_symlink():
            raise CurationTransformError(
                "Relative artifacts require recomputed statistics"
            )
        valid = (
            declared == "deferred"
            and value.get("policy") == POLICY
            and value.get("status") == "deferred"
            and value.get("metadata_sha256") == _metadata_hashes(root)
        )
    except (OSError, ValueError, TypeError, CurationTransformError) as exc:
        raise CurationTransformError("Deferred statistics marker is invalid") from exc
    if not valid:
        raise CurationTransformError(
            "Deferred statistics marker does not match metadata"
        )
    return True


def clear_deferred_statistics(root: Path) -> None:
    from datasetui.dataset_io.files import read_json, write_json

    info_path = root / "meta/info.json"
    info = read_json(info_path)
    if INFO_KEY in info:
        info.pop(INFO_KEY)
        write_json(info_path, info)
    (root / "meta" / MARKER).unlink(missing_ok=True)
    readme = root / "README.md"
    if readme.is_file() and not readme.is_symlink():
        text = readme.read_text()
        if NOTICE_START in text and NOTICE_END in text:
            start = text.index(NOTICE_START)
            end = text.index(NOTICE_END, start) + len(NOTICE_END)
            readme.write_text(text[:start] + text[end:])


def preserve_deferred_statistics(source: Path, destination: Path) -> dict:
    from datasetui.dataset_io.files import read_json, read_regular_bytes, write_json

    stats = source / "meta/stats.json"
    if stats.exists() or stats.is_symlink():
        raw = read_regular_bytes(stats, max_bytes=64 * 1024 * 1024)
        if not isinstance(json.loads(raw), dict):
            raise CurationTransformError("Source statistics must be an object")
    else:
        # An empty metadata object is explicit absence, not invented statistics.
        raw = b"{}\n"
    (destination / "meta/stats.json").write_bytes(raw)
    clear_deferred_statistics(destination)
    (destination / "meta/datasetui_provenance.json").unlink(missing_ok=True)
    info = read_json(destination / "meta/info.json")
    info[INFO_KEY] = "deferred"
    write_json(destination / "meta/info.json", info)
    write_json(
        destination / "meta" / MARKER,
        {
            "policy": POLICY,
            "status": "deferred",
            "metadata_sha256": _metadata_hashes(destination),
        },
    )
    readme = destination / "README.md"
    if readme.is_symlink():
        raise CurationTransformError("Output README is unsafe")
    original = readme.read_text() if readme.exists() else ""
    readme.write_text(
        original + "\n\n" + NOTICE_START + "\n"
        "## Distribution statistics deferred / 분포 통계 미재계산\n\n"
        "`meta/stats.json` retains source metadata for loader compatibility; "
        "it does not describe this output distribution and must not be used for "
        "normalization or fresh statistics aggregation. Compute training norm_stats "
        "from the final selected data and training transforms. Episode lengths, "
        "indices and video ranges describe the edited output.\n\n"
        "분포 통계를 재계산하지 않았습니다. 학습 전 최종 데이터와 학습 설정으로 "
        "norm_stats를 계산하세요.\n" + NOTICE_END + "\n"
    )
    return {
        "policy": POLICY,
        "source": "inherited-unverified",
        "status": "deferred",
        "fallback": False,
    }
