from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from lerobot_dataset_editor.parity.common import evenly_sample
from lerobot_dataset_editor.parity.insights import (
    _action_distribution, _autocorrelation, _jerk, _speed_consistency,
    _episode_rows, _alignment, _episode_length_stats, _variance,
)


def test_episode_jerk_and_speed_cv_match_pinned_semantics() -> None:
    grouped = {0: [[0.0], [1.0], [2.0], [3.0]], 1: [[0.0], [0.0], [0.0], [0.0]]}
    distribution = _action_distribution(grouped, 1)
    actual_jerk = _jerk(grouped, distribution)
    speed = _speed_consistency(grouped)
    golden = json.loads((Path(__file__).parents[1] / "goldens" / "action_insights.json").read_text())
    assert actual_jerk == golden["jerk"]
    assert speed["coefficient_of_variation"] == golden["speed_cv"]
    assert speed["verdict"] == golden["speed_verdict"]
    assert sum(speed["histogram"]["counts"]) == 2


def test_activity_is_per_episode_and_constant_acf_matches_pinned_zero_contribution() -> None:
    grouped = {0: [[0.0], [1.0], [2.0], [3.0]], 1: [[5.0], [5.0], [5.0], [5.0]]}
    distribution = _action_distribution(grouped, 1)["dimensions"][0]
    assert distribution["episode_activity"] == [
        {"episode_index": 0, "p95_delta": 1.0, "active": True},
        {"episode_index": 1, "p95_delta": 0.0, "active": False},
    ]
    assert sum(distribution["histogram"]["counts"]) == 3
    acf = _autocorrelation({0: [[1.0]] * 10}, 1, ["joint"])
    assert acf["series"][0]["values"] == [0.0] * 5
    assert acf["suggested_chunk_size"] == 1


def test_constant_episode_remains_in_aggregate_acf_denominator() -> None:
    varying = [[float(index)] for index in range(10)]
    mixed = _autocorrelation({0: varying, 1: [[4.0]] * 10}, 1, ["joint"])
    varying_only = _autocorrelation({0: varying}, 1, ["joint"])
    assert mixed["series"][0]["values"][0] == varying_only["series"][0]["values"][0] / 2


def test_velocity_full_range_fields_and_speed_bins_match_pinned_golden() -> None:
    grouped = {0: [[0.0], [1.0], [3.0], [6.0]], 1: [[10.0], [10.0], [10.0], [10.0]]}
    dimension = _action_distribution(grouped, 1)["dimensions"][0]
    speed = _speed_consistency(grouped)
    golden = json.loads((Path(__file__).parents[1] / "goldens" / "action_insights.json").read_text())
    assert {key: dimension[key] for key in ("inactive", "discrete", "std", "max_abs", "lo", "hi")} == golden["velocity"]
    assert dimension["histogram"]["edges"][0] == dimension["lo"]
    assert dimension["histogram"]["edges"][-1] == dimension["hi"]
    assert sum(dimension["histogram"]["counts"]) == 3
    assert speed["histogram"]["edges"] == golden["speed_edges"]


def test_episode_length_histogram_matches_pinned_rounding_and_labels() -> None:
    actual = _episode_length_stats({0: 10, 1: 20, 2: 40}, 10)
    assert actual["mean_episode_length"] == 2.33
    assert actual["median_episode_length"] == 2.0
    assert actual["std_episode_length"] == 1.25
    assert actual["episode_length_histogram"] == [
        {"bin_label": "1.0–1.5s", "count": 1},
        {"bin_label": "1.5–2.0s", "count": 0},
        {"bin_label": "2.0–2.5s", "count": 1},
        {"bin_label": "2.5–3.0s", "count": 0},
        {"bin_label": "3.0–3.5s", "count": 0},
        {"bin_label": "3.5–4.0s", "count": 1},
    ]


def test_episode_length_rounding_matches_javascript_math_round_at_half_ties() -> None:
    actual = _episode_length_stats({0: 1}, 8)
    assert actual["all_episode_lengths"][0]["length_seconds"] == 0.13
    assert actual["mean_episode_length"] == 0.13
    assert actual["median_episode_length"] == 0.13
    assert actual["episode_length_histogram"] == [{"bin_label": "0.1s", "count": 1}]


def test_cross_episode_sampling_includes_both_endpoints() -> None:
    table = pa.table({"episode_index": list(range(121)),
                      "action": [[float(index)] for index in range(121)]})
    grouped = _episode_rows(table, "action")
    assert len(grouped) == 120
    assert list(grouped)[0] == 0 and list(grouped)[-1] == 120
    assert evenly_sample(list(range(6)), 3) == [0, 3, 5]


def test_variance_time_bins_use_javascript_half_up_frame_rounding() -> None:
    # Lock the pinned normalized-time frame selection and the resulting
    # population variance at the beginning of the 50-bin sequence.
    ramp = [[float(index)] for index in range(50)]
    zero = [[0.0] for _ in range(50)]
    variance = _variance({0: ramp, 1: zero}, 1)
    assert variance["bins"][1]["variance"] == [0.25]


def test_alignment_suffix_pairs_and_positive_peak_semantics() -> None:
    # State delta follows action delta by two frames. Extra state dimension
    # must not be paired because at least one suffix match exists.
    actions = [[float(index % 3)] for index in range(50)]
    states = [[0.0, 9.0], [0.0, 9.0], *[[float((index - 2) % 3), 9.0] for index in range(2, 50)]]
    result = _alignment({0: actions}, {0: states}, ["joint"], ["joint", "extra"])
    assert len(result) == 1
    assert result[0]["action"] == result[0]["state"] == "joint"
    assert result[0]["correlation"] > 0
