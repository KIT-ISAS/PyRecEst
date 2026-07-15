"""Regressions for range-bearing model Jacobian argument handling."""

import numpy.testing as npt

from pyrecest.backend import array, diag
from pyrecest.models import range_bearing_jacobian, range_bearing_model


def _noise_covariance():
    return diag(array([0.1, 0.2]))


def test_range_bearing_model_jacobian_forwards_runtime_arguments():
    state = array([3.0, 4.0, 13.0, 5.0])
    sensor_position = array([10.0, 1.0])
    position_indices = (2, 3)
    model = range_bearing_model(_noise_covariance())

    actual = model.jacobian(
        state,
        sensor_position=sensor_position,
        position_indices=position_indices,
    )
    expected = range_bearing_jacobian(
        state,
        sensor_position=sensor_position,
        position_indices=position_indices,
    )

    npt.assert_allclose(actual, expected)


def test_range_bearing_model_snapshots_mutable_position_indices_for_jacobian():
    position_indices = [0, 1]
    state = array([3.0, 4.0, 13.0, 5.0])
    model = range_bearing_model(
        _noise_covariance(),
        position_indices=position_indices,
    )

    position_indices[:] = [2, 3]

    npt.assert_allclose(
        model.jacobian(state),
        range_bearing_jacobian(state, position_indices=(0, 1)),
    )
