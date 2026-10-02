from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from datasetui.relative import actions as relative_actions
from datasetui.relative.actions import (
    compute_relative_action_profile,
    dimension_options,
)
from datasetui.transform_errors import CurationTransformError



def _info(
    action_names: list[str] | dict[str, list[str]] | None = None,
    state_names: list[str] | dict[str, list[str]] | None = None,
    *,
    action_dtype: str = "float32",
    state_dtype: str = "float32",
) -> dict:
    action_names = (
        action_names if action_names is not None else ["joint1", "joint10", "gripper"]
    )
    state_names = (
        state_names if state_names is not None else ["joint1", "joint10", "gripper"]
    )

    def width(names):
        if isinstance(names, dict):
            return sum(len(values) for values in names.values())
        return len(names)

    return {
        "features": {
            "action": {
                "dtype": action_dtype,
                "shape": [width(action_names)],
                "names": action_names,
            },
            "observation.state": {
                "dtype": state_dtype,
                "shape": [width(state_names)],
                "names": state_names,
            },
        }
    }


def _episode(actions: list[list[float]], states: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action": [np.asarray(row, dtype=np.float32) for row in actions],
            "observation.state": [np.asarray(row, dtype=np.float32) for row in states],
        }
    )


@pytest.fixture(autouse=True)
def official_function(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[int, ...], tuple[bool, ...]]] = []

    def official(actions, state, mask):
        calls.append((tuple(actions.shape), tuple(mask)))
        mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
        output = actions.clone()
        output[..., : len(mask)] -= state[..., : len(mask)].unsqueeze(1) * mask_t
        return output

    official.calls = calls
    monkeypatch.setattr(
        relative_actions, "_load_official_to_relative_actions", lambda: official
    )


def _independent_values(
    episodes: list[pd.DataFrame], chunk_size: int, mask: list[bool]
) -> np.ndarray:
    result = []
    mask_array = np.asarray(mask, dtype=np.float32)
    for episode in episodes:
        action = np.stack(episode["action"].tolist())
        state = np.stack(episode["observation.state"].tolist())
        for start in range(len(action) - chunk_size + 1):
            chunk = action[start : start + chunk_size].copy()
            chunk -= state[start][None, :] * mask_array[None, :]
            result.append(chunk)
    return np.concatenate(result, axis=0)


def test_profile_uses_exact_selection_and_exact_chunk_statistics() -> None:
    episodes = [
        _episode(
            [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]],
            [[0.5, 5, 50], [1.5, 15, 150], [2.5, 25, 250], [3.5, 35, 350]],
        ),
        _episode(
            [[6, 60, 600], [7, 70, 700], [8, 80, 800]],
            [[5.5, 55, 550], [6.5, 65, 650], [7.5, 75, 750]],
        ),
    ]
    originals = [episode.copy(deep=True) for episode in episodes]
    profile = compute_relative_action_profile(
        _info(),
        episodes,
        {"enabled": True, "dimensions": ["joint10"], "chunk_size": 2},
    )

    expected = _independent_values(episodes, 2, [False, True, False]).astype(np.float64)
    assert profile["mask"] == [False, True, False]
    assert profile["dimensions"] == ["joint10"]
    assert profile["absolute_dimensions"] == ["joint1", "gripper"]
    assert profile["valid_chunks"] == 5
    assert profile["statistics"]["count"] == [10]
    for key, expected_value in {
        "min": np.min(expected, axis=0),
        "max": np.max(expected, axis=0),
        "mean": np.mean(expected, axis=0),
        "std": np.std(expected, axis=0),
        "q01": np.quantile(expected, 0.01, axis=0),
        "q10": np.quantile(expected, 0.10, axis=0),
        "q50": np.quantile(expected, 0.50, axis=0),
        "q90": np.quantile(expected, 0.90, axis=0),
        "q99": np.quantile(expected, 0.99, axis=0),
    }.items():
        np.testing.assert_allclose(profile["statistics"][key], expected_value)
    assert profile["processor_config"] == {
        "enabled": True,
        "action_names": [
            "datasetui_dim_0000",
            "datasetui_dim_0001",
            "datasetui_dim_0002",
        ],
        "exclude_joints": ["datasetui_dim_0000", "datasetui_dim_0002"],
    }
    for before, after in zip(originals, episodes):
        for key in ("action", "observation.state"):
            for expected_row, actual_row in zip(before[key], after[key]):
                np.testing.assert_array_equal(actual_row, expected_row)


def test_chunks_do_not_cross_episode_boundaries_and_short_episodes_are_reported() -> (
    None
):
    episodes = [
        _episode([[1, 10, 100], [2, 20, 200]], [[1, 1, 1], [2, 2, 2]]),
        _episode([[9, 90, 900]], [[9, 9, 9]]),
        _episode(
            [[3, 30, 300], [4, 40, 400], [5, 50, 500]],
            [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
        ),
    ]
    profile = compute_relative_action_profile(
        _info(),
        episodes,
        {"enabled": True, "dimensions": ["joint1"], "chunk_size": 2},
    )
    assert profile["valid_chunks"] == 3
    assert profile["episodes_total"] == 3
    assert profile["episodes_used"] == 2
    assert profile["episodes_skipped_short"] == 1
    assert profile["statistics"]["count"] == [6]


def test_unchecked_dimensions_remain_bit_exact() -> None:
    episode = _episode(
        [[1.125, 10.25, 100.5], [2.125, 20.25, 200.5]],
        [[0.5, 5, 50], [1.5, 15, 150]],
    )
    profile = compute_relative_action_profile(
        _info(),
        [episode],
        {"enabled": True, "dimensions": ["joint10"], "chunk_size": 1},
    )
    expected = np.stack(episode["action"].tolist()).astype(np.float64)
    np.testing.assert_array_equal(
        np.asarray(profile["statistics"]["min"])[[0, 2]],
        expected.min(axis=0)[[0, 2]],
    )
    np.testing.assert_array_equal(
        np.asarray(profile["statistics"]["max"])[[0, 2]],
        expected.max(axis=0)[[0, 2]],
    )


def test_dimension_options_accepts_dict_names_and_only_same_index_names() -> None:
    info = _info(
        {"joints": ["shoulder", "elbow", "gripper"]},
        {"joints": ["shoulder", "other", "gripper"]},
    )
    assert dimension_options(info) == ["shoulder", "gripper"]


def test_multiple_name_groups_and_multiaxis_features_are_rejected() -> None:
    grouped = _info()
    grouped["features"]["action"]["names"] = {
        "arm": ["joint1"],
        "hand": ["joint10", "gripper"],
    }
    with pytest.raises(CurationTransformError, match="explicit action dimension names"):
        dimension_options(grouped)

    matrix = _info()
    matrix["features"]["action"]["shape"] = [1, 3]
    with pytest.raises(CurationTransformError, match="shape is invalid"):
        dimension_options(matrix)


def test_disabled_profile_does_not_load_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        relative_actions,
        "_load_official_to_relative_actions",
        lambda: pytest.fail("runtime must not be loaded"),
    )
    assert compute_relative_action_profile({}, [], {"enabled": False}) == {
        "enabled": False,
        "dimensions": [],
        "statistics": {},
    }


def test_runtime_gate_failure_has_no_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail():
        raise CurationTransformError("identity gate failed")

    monkeypatch.setattr(relative_actions, "_load_official_to_relative_actions", fail)
    with pytest.raises(CurationTransformError, match="identity gate failed"):
        compute_relative_action_profile(
            _info(),
            [_episode([[1, 2, 3]], [[0, 0, 0]])],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": 1},
        )


@pytest.mark.parametrize(
    ("info", "message"),
    [
        (_info(["joint", "joint", "gripper"]), "must be unique"),
        (_info(action_dtype="float64"), "requires float32"),
        (
            _info(["joint1", "joint10"], ["joint1", "joint10", "gripper"]),
            "equal action and state widths",
        ),
    ],
)
def test_invalid_metadata_is_rejected(info: dict, message: str) -> None:
    with pytest.raises(CurationTransformError, match=message):
        compute_relative_action_profile(
            info,
            [_episode([[1, 2, 3]], [[0, 0, 0]])],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": 1},
        )


@pytest.mark.parametrize(
    "bad_value", [None, np.asarray([1, np.nan, 3], dtype=np.float32)]
)
def test_null_and_nonfinite_values_are_rejected(bad_value) -> None:
    episode = _episode([[1, 2, 3]], [[0, 0, 0]])
    episode.at[0, "action"] = bad_value
    with pytest.raises(CurationTransformError, match="null|NaN or infinity"):
        compute_relative_action_profile(
            _info(),
            [episode],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": 1},
        )


def test_unknown_and_misaligned_names_are_rejected() -> None:
    episode = _episode([[1, 2, 3]], [[0, 0, 0]])
    with pytest.raises(CurationTransformError, match="Unknown"):
        compute_relative_action_profile(
            _info(),
            [episode],
            {"enabled": True, "dimensions": ["missing"], "chunk_size": 1},
        )
    with pytest.raises(CurationTransformError, match="same index"):
        compute_relative_action_profile(
            _info(state_names=["joint10", "joint1", "gripper"]),
            [episode],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": 1},
        )


def test_no_valid_chunks_reports_short_episode_counts() -> None:
    with pytest.raises(
        CurationTransformError,
        match=r"episodes=1, skipped_short=1, chunk_size=3",
    ):
        compute_relative_action_profile(
            _info(),
            [_episode([[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [1, 1, 1]])],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": 3},
        )


@pytest.mark.parametrize("chunk_size", [0, 1025, True])
def test_chunk_size_bounds_are_enforced(chunk_size) -> None:
    with pytest.raises(CurationTransformError, match="between 1 and 1024"):
        compute_relative_action_profile(
            _info(),
            [_episode([[1, 2, 3]], [[0, 0, 0]])],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": chunk_size},
        )


def test_scratch_size_cap_rejects_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASETUI_RELATIVE_MAX_SCRATCH_BYTES", "23")
    monkeypatch.setattr(
        relative_actions,
        "_matrix",
        lambda *args, **kwargs: pytest.fail("arrays must not load before cap check"),
    )
    with pytest.raises(CurationTransformError, match="exceed the scratch limit"):
        compute_relative_action_profile(
            _info(),
            [_episode([[1, 2, 3]], [[0, 0, 0]])],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": 1},
        )


def test_scratch_free_space_requires_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        relative_actions.shutil,
        "disk_usage",
        lambda _: usage(1000, 900, 100),
    )
    with pytest.raises(CurationTransformError, match="need more scratch space"):
        compute_relative_action_profile(
            _info(),
            [_episode([[1, 2, 3]], [[0, 0, 0]])],
            {"enabled": True, "dimensions": ["joint1"], "chunk_size": 1},
        )


def test_configured_scratch_directory_is_cleaned(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATASETUI_RELATIVE_SCRATCH_DIR", str(tmp_path))
    compute_relative_action_profile(
        _info(),
        [_episode([[1, 2, 3]], [[0, 0, 0]])],
        {"enabled": True, "dimensions": ["joint1"], "chunk_size": 1},
    )
    assert list(tmp_path.iterdir()) == []


def test_lazy_episode_sequence_is_processed_one_episode_at_a_time() -> None:
    source = [
        _episode([[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [1, 1, 1]]),
        _episode([[7, 8, 9], [10, 11, 12]], [[2, 2, 2], [3, 3, 3]]),
    ]

    class LazyEpisodes(Sequence[pd.DataFrame]):
        def __init__(self) -> None:
            self.accesses: list[int] = []

        def __len__(self) -> int:
            return len(source)

        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError(
                    "relative calculation must not materialize a slice"
                )
            self.accesses.append(index)
            return source[index].copy(deep=True)

    episodes = LazyEpisodes()
    profile = compute_relative_action_profile(
        _info(),
        episodes,
        {"enabled": True, "dimensions": ["joint1"], "chunk_size": 2},
    )
    assert profile["valid_chunks"] == 2
    assert profile["statistics"]["count"] == [4]
    assert episodes.accesses == [0, 1, 0, 1]
