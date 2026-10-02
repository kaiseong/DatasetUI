"""Merge must retain upstream statistics without a second data/video pass."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from datasetui.official import operations




def test_only_merge_uses_official_aggregated_statistics():
    merge = operations.provenance("merge_datasets")
    assert merge["statistics_policy"] == "lerobot-official-aggregate-v1"
    assert (
        operations.provenance("split_dataset")["statistics_policy"]
        == "lerobot-official-aggregate-v1"
    )


def test_merge_retains_official_global_and_episode_stats(tmp_path, monkeypatch):
    from datasetui import output_metadata
    from datasetui.statistics import output as output_statistics
    from datasetui.merge import schema as merge_schema

    sources = []
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "meta").mkdir(parents=True)
        (root / "meta/info.json").write_text(json.dumps({"features": {}}))
        (root / "original.mp4").write_bytes(name.encode())
        sources.append(SimpleNamespace(root=root))
    before = [operations.inventory(source.root) for source in sources]
    destination = tmp_path / "merged"
    # Deliberately unlike exact merged quantiles: do not silently overwrite them.
    stats_bytes = b'{"action":{"q01":[-100],"q99":[100]}}\n'
    episode_bytes = b"official episode metadata sentinel"
    calls = []

    def official_merge(loaded, repo_id, *, output_dir, **options):
        calls.append(options)
        (output_dir / "meta/episodes").mkdir(parents=True)
        (output_dir / "data").mkdir()
        (output_dir / "meta/info.json").write_text(
            json.dumps({"features": {}}), encoding="utf-8"
        )
        (output_dir / "meta/stats.json").write_bytes(stats_bytes)
        pd.DataFrame({"sentinel": [episode_bytes]}).to_parquet(
            output_dir / "meta/episodes/file.parquet", index=False
        )
        pd.DataFrame({"index": [0]}).to_parquet(
            output_dir / "data/file.parquet", index=False
        )
        for index, source in enumerate(sources):
            (output_dir / f"video-{index}.mp4").write_bytes(
                (source.root / "original.mp4").read_bytes()
            )
        return [None] * 4

    def forbid_recomputation(*args, **kwargs):
        pytest.fail("official merge must not recompute dataset/episode statistics")

    monkeypatch.setattr(
        operations,
        "require_runtime",
        lambda: (object, SimpleNamespace(merge_datasets=official_merge)),
    )
    monkeypatch.setattr(
        operations,
        "_load_source",
        lambda *args: SimpleNamespace(
            meta=SimpleNamespace(robot_type="rby1", episodes=[{"length": 2}])
        ),
    )
    monkeypatch.setattr(
        merge_schema, "inspect_compatible_data_schema", lambda roots: None
    )
    monkeypatch.setattr(
        merge_schema, "restore_merged_data_schema", lambda **kwargs: {"unchanged": True}
    )
    monkeypatch.setattr(output_metadata, "copy_modality_metadata", lambda *args: None)
    monkeypatch.setattr(
        output_statistics, "write_output_statistics", forbid_recomputation
    )
    result = operations._merge_private(
        sources=sources, destination=destination, robot_type="rby1"
    )

    assert calls == [{"concatenate_videos": False, "concatenate_data": False}]
    assert (destination / "meta/stats.json").read_bytes() == stats_bytes
    assert pd.read_parquet(destination / "meta/episodes/file.parquet").iloc[0][
        "sentinel"
    ] == episode_bytes
    marker = json.loads(
        (destination / "meta/datasetui_provenance.json").read_text(encoding="utf-8")
    )
    assert marker["official_function"] == "merge_datasets"
    assert [operations.inventory(source.root) for source in sources] == before
    assert result["processing"]["statistics_policy"] == "lerobot-official-aggregate-v1"
    assert len(result["lineage"]) == 2
