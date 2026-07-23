from __future__ import annotations

import numpy as np
import pytest
from pyrecest.models import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    diagonal_measurement_covariance,
    selection_matrix,
)


def _object_array_with_nested_scalar(value, shape):
    array = np.empty(shape, dtype=object)
    array.flat[0] = np.asarray(value)
    return array


def test_nested_complex_standard_deviation_is_rejected() -> None:
    stds = _object_array_with_nested_scalar(1.0 + 2.0j, (1,))

    with pytest.raises(ValueError, match="real numeric values"):
        diagonal_measurement_covariance(stds)
    with pytest.raises(ValueError, match="real numeric values"):
        MaskedLinearMeasurementModel(state_dim=1, observed_dims=[0], stds=stds)
    with pytest.raises(ValueError, match="real numeric values"):
        WeakDimensionMeasurementModel(np.eye(1), stds=stds)


def test_nested_complex_measurement_covariance_is_rejected() -> None:
    covariance = _object_array_with_nested_scalar(1.0 + 2.0j, (1, 1))

    with pytest.raises(ValueError, match="measurement_noise_cov"):
        MaskedLinearMeasurementModel(
            state_dim=1,
            observed_dims=[0],
            measurement_noise_cov=covariance,
        )
    with pytest.raises(ValueError, match="measurement_noise_cov"):
        WeakDimensionMeasurementModel(
            np.eye(1),
            measurement_noise_cov=covariance,
        )


def test_nested_boolean_dimensions_are_rejected() -> None:
    state_dim = _object_array_with_nested_scalar(True, ())
    observed_dim = _object_array_with_nested_scalar(True, ())

    with pytest.raises(ValueError, match="state_dim must be a nonnegative integer"):
        selection_matrix(state_dim, [0])
    with pytest.raises(ValueError, match="observed_dims must be a nonnegative integer"):
        selection_matrix(2, [observed_dim])
