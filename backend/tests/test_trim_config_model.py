import math

import pytest
from pydantic import ValidationError

from datasetui.models import TrimConfig


def test_existing_trim_payload_defaults_to_legacy_motion() -> None:
    config = TrimConfig.model_validate({"enabled": True})

    assert config.method == "legacy_motion"
    assert config.state_epsilon == 0.0005


def test_stationary_trim_accepts_positive_finite_state_epsilon() -> None:
    config = TrimConfig.model_validate(
        {"enabled": True, "method": "stationary", "state_epsilon": 0.01}
    )

    assert config.method == "stationary"
    assert config.state_epsilon == 0.01


def test_stationary_trim_rejects_dimension_selection() -> None:
    with pytest.raises(ValidationError, match="every observation.state dimension"):
        TrimConfig.model_validate(
            {
                "method": "stationary",
                "dimensions": ["joint_0"],
            }
        )


@pytest.mark.parametrize("value", [0, -0.01, math.inf, -math.inf, math.nan])
def test_stationary_trim_rejects_non_positive_or_non_finite_epsilon(
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        TrimConfig.model_validate({"method": "stationary", "state_epsilon": value})
