import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from datasetui.models import TrimConfig
from datasetui.transforms import _trim_bounds


def bounds(**changes):
    increments = np.zeros(20)
    increments[3:5] = 1
    increments[8:13] = 1
    increments[16:17] = 1
    values = [[float(x)] for x in np.r_[0, np.cumsum(increments)]]
    data = pd.DataFrame({"action": values, "observation.state": values})
    config = TrimConfig(
        enabled=True, threshold=0.01, hold_time_s=0.1, margin_s=0, **changes
    ).model_dump()
    return _trim_bounds(data, {}, 10, config, 0)


def test_legacy_symmetric_timing():
    assert bounds() == (4, 18, "motion")


def test_margins_are_independent_and_zero_is_not_a_fallback():
    assert bounds(start_margin_s=0.2, end_margin_s=0) == (2, 18, "motion")
    assert bounds(start_margin_s=0, end_margin_s=0.2) == (4, 20, "motion")
    assert bounds(start_margin_s=60, end_margin_s=60) == (0, 21, "motion")


def test_hold_times_are_independent_and_keep_middle_pauses():
    assert bounds(start_hold_time_s=0.3, end_hold_time_s=0.1) == (9, 18, "motion")
    assert bounds(start_hold_time_s=0.1, end_hold_time_s=0.3) == (4, 14, "motion")


def test_unmatched_side_preserves_its_edge():
    assert bounds(start_hold_time_s=3, end_hold_time_s=0.3) == (0, 14, "motion")
    assert bounds(start_hold_time_s=0.3, end_hold_time_s=3) == (9, 21, "motion")
    assert bounds(start_hold_time_s=3, end_hold_time_s=3) == (
        0,
        21,
        "no_sustained_motion",
    )


def test_manual_override_still_wins():
    assert bounds(episode_overrides={0: {"start_frame": 2, "end_frame": 19}}) == (
        2,
        19,
        "manual",
    )


@pytest.mark.parametrize(
    "field", ["start_hold_time_s", "end_hold_time_s", "start_margin_s", "end_margin_s"]
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 61])
def test_invalid_side_timing_rejected(field, value):
    with pytest.raises(ValidationError):
        TrimConfig(**{field: value})
