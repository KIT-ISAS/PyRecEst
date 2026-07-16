from __future__ import annotations

import numpy as np
import pytest
from pyrecest.models import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
    diagonal_measurement_covariance,
)


@pytest.mark.parametrize(
    "std",
    [np.finfo(float).max, np.nextafter(0.0, 1.0)],
    ids=["overflow", "underflow"],
)
def test_measurement_covariance_rejects_unrepresentable_variances(std: float) -> None:
    with pytest.raises(ValueError, match="finite positive variances"):
        diagonal_measurement_covariance([std])
    with pytest.raises(ValueError, match="finite positive variances"):
        block_diag_measurement_covariance(trusted_std=[std])


def test_measurement_models_reject_unrepresentable_variances() -> None:
    std = np.finfo(float).max

    with pytest.raises(ValueError, match="finite positive variances"):
        MaskedLinearMeasurementModel(state_dim=1, observed_dims=[0], stds=[std])
    with pytest.raises(ValueError, match="finite positive variances"):
        WeakDimensionMeasurementModel(np.eye(1), stds=[std])
