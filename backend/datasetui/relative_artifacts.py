"""Read and verify explicit mixed relative/absolute training artifacts."""

import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from datasetui.transform_errors import CurationTransformError



def load_relative_processors(dataset_root: Path, *, chunk_size: int):
    """Insert relative BEFORE normalization; paired absolute AFTER unnormalization.

    Replace existing relative/absolute steps, never apply the conversion twice.
    The caller must use this dataset's meta/stats.json for normalization.
    """
    from datasetui.lerobot_runtime import require_runtime
    from datasetui.dataset_io.files import read_regular_bytes

    require_runtime()
    from lerobot.processor import (
        DataProcessorPipeline,
        RelativeActionsProcessorStep,
        AbsoluteActionsProcessorStep,
    )

    root = Path(dataset_root)
    profile = read_relative_profile(root)
    if profile is None:
        raise ValueError("Dataset has no Relative training profile")
    info = json.loads(
        read_regular_bytes(root / "meta/info.json", max_bytes=16 * 1024 * 1024)
    )
    from datasetui.relative_actions import dimension_options

    compatible_names = set(dimension_options(info))
    if not set(profile["dimensions"]).issubset(compatible_names):
        raise ValueError("Relative action dimensions do not match dataset metadata")
    stats = json.loads(
        read_regular_bytes(root / "meta/stats.json", max_bytes=64 * 1024 * 1024)
    )
    absolute_stats = json.loads(
        read_regular_bytes(
            root / "meta/stats.absolute.json", max_bytes=64 * 1024 * 1024
        )
    )
    if (
        not isinstance(stats, dict)
        or not isinstance(absolute_stats, dict)
        or "action" not in absolute_stats
        or stats.get("action") != profile.get("statistics")
    ):
        raise ValueError("Relative training statistics do not match the profile")
    if (
        type(chunk_size) is not int
        or chunk_size != profile["chunk_size"]
        or profile.get("action_horizon") != chunk_size
        or profile.get("stored_action") != "absolute"
        or profile.get("enabled") is not True
    ):
        raise ValueError("Relative training profile or chunk size mismatch")
    names = info["features"]["action"]["names"]
    states = info["features"]["observation.state"]["names"]
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    if isinstance(states, dict) and len(states) == 1:
        states = next(iter(states.values()))
    if (
        not isinstance(names, list)
        or not isinstance(states, list)
        or len(names) != len(states)
        or len(names) != len(set(names))
        or names != profile["action_names"]
    ):
        raise ValueError("Relative action names no longer match dataset metadata")
    selected = set(profile["dimensions"])
    mask = [name in selected for name in names]
    if (
        not selected
        or not selected.issubset(names)
        or mask != profile["mask"]
        or any(names[i] != states[i] for i, enabled in enumerate(mask) if enabled)
    ):
        raise ValueError("Relative action/state mapping or mask mismatch")
    aliases = [f"datasetui_dim_{index:04d}" for index in range(len(names))]
    config = {
        "enabled": True,
        "action_names": aliases,
        "exclude_joints": [name for name, enabled in zip(aliases, mask) if not enabled],
    }
    if config != profile["processor_config"]:
        raise ValueError("Relative processor configuration mismatch")
    expected = _processor_artifacts(profile)
    _verify_processor_artifacts(root, profile)
    # from_pretrained reads JSON again. Give it private canonical copies so an
    # external file replacement cannot introduce a dynamic class after validation.
    with tempfile.TemporaryDirectory(
        prefix="datasetui-relative-processors-"
    ) as scratch:
        private = Path(scratch)
        for filename, value in expected.items():
            (private / filename).write_text(json.dumps(value), encoding="utf-8")
        pre = DataProcessorPipeline.from_pretrained(
            private,
            config_filename="datasetui_relative_preprocessor.json",
        )
        if len(pre.steps) != 1 or not isinstance(
            pre.steps[0], RelativeActionsProcessorStep
        ):
            raise ValueError("Unexpected relative processor pipeline")
        relative = pre.steps[0]
        if relative._build_mask(len(names)) != mask:
            raise ValueError(
                "Installed LeRobot cannot reproduce the exact selected mask"
            )
        post = DataProcessorPipeline.from_pretrained(
            private,
            config_filename="datasetui_relative_postprocessor.json",
            overrides={"absolute_actions_processor": {"relative_step": relative}},
        )
    if (
        len(post.steps) != 1
        or not isinstance(post.steps[0], AbsoluteActionsProcessorStep)
        or post.steps[0].relative_step is not relative
    ):
        raise ValueError("Relative/absolute processors were not paired")
    # Re-read after loading as a consistency guard; never execute dataset code.
    _verify_processor_artifacts(root, profile)
    absolute = post.steps[0]
    return relative, absolute


def _processor_artifacts(profile: dict) -> dict:
    return {
        "datasetui_relative_preprocessor.json": {
            "name": "DatasetUI relative action preprocessor",
            "steps": [
                {
                    "registry_name": "relative_actions_processor",
                    "config": profile["processor_config"],
                }
            ],
        },
        "datasetui_relative_postprocessor.json": {
            "name": "DatasetUI relative action postprocessor",
            "steps": [
                {
                    "registry_name": "absolute_actions_processor",
                    "config": {"enabled": True},
                }
            ],
        },
    }


def _processor_fingerprint(artifacts: dict) -> str:
    return hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _verify_processor_artifacts(root: Path, profile: dict) -> None:
    from datasetui.dataset_io.files import read_regular_bytes

    expected = _processor_artifacts(profile)
    if profile.get("processor_sha256") != _processor_fingerprint(expected):
        raise CurationTransformError("Relative processor fingerprint differs")
    for filename, value in expected.items():
        actual = json.loads(
            read_regular_bytes(root / "meta" / filename, max_bytes=4 * 1024 * 1024)
        )
        if actual != value:
            raise CurationTransformError(
                f"Relative processor artifact differs: {filename}"
            )


def write_training_instructions(root: Path) -> None:
    """Export declarative, allowlisted official processor configurations."""
    from datasetui.dataset_io.files import write_json

    profile = read_relative_profile(root)
    if profile is None:
        raise CurationTransformError("Missing Relative training profile")
    artifacts = _processor_artifacts(profile)
    for filename, value in artifacts.items():
        write_json(root / "meta" / filename, value)
    profile["processor_sha256"] = _processor_fingerprint(artifacts)
    write_json(root / "meta/relative_action.json", profile)
    (root / "RELATIVE_TRAINING.md").write_text(
        """# Relative training profile

This is a training-profile dataset, not a dataset with permanently subtracted actions.
Stored Parquet actions and observations remain absolute. `meta/stats.json` contains
mixed chunk-relative/absolute action normalization statistics; all other feature
statistics remain absolute. `meta/stats.absolute.json` preserves raw-data statistics.

Do NOT train in absolute-action mode with the mixed action statistics. The dataset
does not automatically modify any training project or policy configuration.

## Official LeRobot processors (verified against pinned 0.6.2)

```python
from pathlib import Path
from datasetui.relative_artifacts import load_relative_processors

root = Path("/path/to/this/dataset")
relative_step, absolute_step = load_relative_processors(
    root, chunk_size=YOUR_POLICY_CHUNK_SIZE,
)
```

Wire `relative_step` BEFORE action normalization in the training/inference
preprocessor, and the paired `absolute_step` AFTER action unnormalization in the
inference postprocessor. Replace any existing relative/absolute steps; do not apply
relative conversion twice. Use this dataset's `meta/stats.json` for normalization.
The loader requires the chunk size recorded in `meta/relative_action.json`.
It is trusted DatasetUI package code and uses the pinned LeRobot runtime. Dataset
files contain JSON configuration only, not executable Python. Install the matching
DatasetUI processing environment to use this helper. Do not pass these one-step
artifacts as the policy's full pretrained pre/postprocessor pipelines: they do not
contain tokenizer, normalization, batching or device steps.

Selected dimensions use action[t+k] - state[t]; unchecked dimensions are unchanged.
Equal-length internal aliases prevent LeRobot substring exclusion rules from
accidentally affecting similarly named joints. They do not rename dataset columns.
Do not substitute an unverified `relative_exclude_joints` CLI list for this loader.

Statistics include every full chunk within an episode, matching the official
relative-statistics convention. Incomplete trailing chunks are not included.
Exact global quantiles are retained instead of approximate histogram quantiles.

This is a LeRobot processor contract, not automatic compatibility with a separate
OpenPI or NVIDIA GR00T training implementation. Configure and verify that consumer
separately; its horizon, padding, dimension mapping and normalization must match.

For merge or v2.1 conversion, process the original absolute dataset first, then
apply Relative to the output. Re-curation of this profile requires explicitly
enabling Relative so statistics are recalculated for the new selection.
""",
        encoding="utf-8",
    )


def reject_relative_profile(root: Path, *, operation: str) -> None:
    """Do not silently turn a training-profile dataset back into absolute stats."""
    if read_relative_profile(root) is not None:
        raise CurationTransformError(
            f"{operation} would discard the Relative training profile. "
            "Use the original absolute dataset for merge/conversion, then apply "
            "Relative with the desired dimensions and chunk size to the result."
        )


def read_relative_profile(root: Path) -> dict | None:
    from datasetui.dataset_io.files import read_regular_bytes

    path = root / "meta/relative_action.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        profile = json.loads(read_regular_bytes(path, max_bytes=4 * 1024 * 1024))
        if (
            not isinstance(profile, dict)
            or type(profile.get("format_version")) is not int
            or profile.get("format_version") != 1
            or profile.get("enabled") is not True
            or profile.get("stored_action") != "absolute"
            or profile.get("training_normalization") is not True
        ):
            raise ValueError("Invalid relative action artifact contract")
        from datasetui.models import RelativeActionConfig

        # Missing horizon is not a legacy default when loading a training artifact.
        config = RelativeActionConfig(
            enabled=True,
            dimensions=profile.get("dimensions"),
            chunk_size=profile.get("chunk_size"),
        )
        if profile.get("action_horizon") != config.chunk_size:
            raise ValueError("Relative chunk size and action horizon differ")
    except (OSError, ValueError, TypeError) as exc:
        raise CurationTransformError(f"Invalid relative action profile: {exc}") from exc
    return profile


def recompute_relative_artifact(
    root: Path, info: dict, profile: dict, *, on_progress=None
) -> tuple[dict, dict]:
    from datasetui.relative_actions import compute_relative_action_profile
    from datasetui.dataset_io.source import DatasetSource

    source = DatasetSource(root, info)
    episodes = DatasetEpisodes(source)
    actual = compute_relative_action_profile(
        info,
        episodes,
        {
            "enabled": True,
            "dimensions": profile["dimensions"],
            "chunk_size": profile["chunk_size"],
        },
        on_progress=on_progress,
    )
    # Never trust a stale/tampered mask or processor configuration independently
    # from selected names, physical column order and the chunk horizon.
    for field in (
        "mask",
        "action_names",
        "absolute_dimensions",
        "processor_config",
        "stored_action",
        "training_normalization",
        "statistics_scope",
        "engine",
        "official_function",
        "upstream_commit",
    ):
        if field not in actual or profile.get(field) != actual[field]:
            raise CurationTransformError(f"Relative action profile differs: {field}")
    _verify_processor_artifacts(root, profile)
    from datasetui.exact_statistics import recompute_numeric_statistics

    return actual["statistics"], recompute_numeric_statistics(
        root, on_progress=on_progress
    )


class DatasetEpisodes(Sequence):
    """A repeatable episode view that never retains an entire dataset in memory."""

    def __init__(self, source, indices=None):
        self.source = source
        self.indices = (
            list(indices) if indices is not None else sorted(source.episode_metadata)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return DatasetEpisodes(self.source, self.indices[index])
        return self.source.episode(self.indices[index])[0]
