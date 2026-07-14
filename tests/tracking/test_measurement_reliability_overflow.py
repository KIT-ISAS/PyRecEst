from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import (
    apply_measurement_reliability,
    reliability_to_covariance_scale,
)


def test_large_exponent_respects_max_scale_before_power_underflow() -> None:
    scale = reliability_to_covariance_scale(
        0.5,
        exponent=1.0e6,
        max_scale=7.0,
    )
    result = apply_measurement_reliability(
        np.eye(2),
        reliability=0.5,
        exponent=1.0e6,
        max_scale=7.0,
    )

    assert scale == 7.0
    assert result.covariance_scale == 7.0
    assert np.allclose(result.covariance, 7.0 * np.eye(2))


def test_large_uncapped_scale_raises_clear_overflow_error() -> None:
    with pytest.raises(ValueError, match="covariance scale overflows"):
        reliability_to_covariance_scale(0.5, exponent=1.0e6)
