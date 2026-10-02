"""Recompute new output statistics; never edit source/training normalization."""

from __future__ import annotations

from pathlib import Path

from datasetui.statistics.episode import write_episode_statistics
from datasetui.statistics.exact import recompute_numeric_statistics
from datasetui.statistics.visual import recompute_visual_statistics_with_episodes
from datasetui.transform_errors import CurationTransformError


STATISTICS_POLICY = "exact-global-numeric-sampled-rgb-v1"


def write_output_statistics(root: Path, *, on_progress=None) -> dict:
    from datasetui.official.operations import enabled
    from datasetui.statistics.deferred import clear_deferred_statistics

    clear_deferred_statistics(root)

    provenance = root / "meta/datasetui_provenance.json"
    if provenance.exists() or provenance.is_symlink():
        if provenance.is_symlink() or not provenance.is_file():
            raise CurationTransformError("Statistics provenance path is unsafe")
        provenance.unlink()
    stats = recompute_numeric_statistics(root, on_progress=on_progress)
    visual_stats, visual_stats_by_episode = (
        recompute_visual_statistics_with_episodes(root, on_progress=on_progress)
    )
    stats.update(visual_stats)
    write_episode_statistics(
        root,
        on_progress=on_progress,
        visual_stats_by_episode=visual_stats_by_episode,
    )
    if enabled():
        from datasetui.official.runtime import require_runtime

        require_runtime()
        from lerobot.datasets.io_utils import write_stats

        write_stats(stats, root)
    else:
        from datasetui.dataset_io.files import write_json

        write_json(root / "meta/stats.json", stats)
    return stats
