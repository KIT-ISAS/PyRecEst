# pylint: disable=no-name-in-module,no-member
"""Tests for robust linear-Gaussian helper functions."""

import numpy as np
import pytest
from pyrecest.backend import allclose, array
from pyrecest.filters._linear_gaussian import (
    huber_covariance_scale,
    linear_gaussian_update,
    normalized_innovation_squared,
    student_t_covariance_scale,
)


def test_huber_covariance_scale_leaves_inliers_unchanged():
    assert allclose(huber_covariance_scale(1.0, huber_threshold=2.0), 1.0)


def test_huber_covariance_scale_downweights_outliers():
    assert allclose(huber_covariance_scale(9.0, huber_threshold=2.0), 1.5)


def test_huber_covariance_scale_vectorizes_over_nis():
    scale = huber_covariance_scale(array([1.0, 4.0, 16.0]), huber_threshold=2.0)

    assert allclose(scale, array([1.0, 1.0, 2.0]))


def test_huber_covariance_scale_rejects_nonpositive_threshold():
    with pytest.raises(ValueError, match="positive"):
        huber_covariance_scale(1.0, huber_threshold=0.0)


@pytest.mark.parametrize(
    "invalid_nis",
    [-1.0, array([0.0, -1.0])],
)
def test_huber_covariance_scale_rejects_negative_nis(invalid_nis):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        huber_covariance_scale(invalid_nis, huber_threshold=2.0)


@pytest.mark.parametrize(
    "invalid_nis",
    [float("nan"), float("inf"), -float("inf")],
)
def test_huber_covariance_scale_rejects_nonfinite_nis(invalid_nis):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        huber_covariance_scale(invalid_nis, huber_threshold=2.0)


@pytest.mark.parametrize(
    "invalid_nis",
    [True, np.bool_(True), array([True])],
)
def test_huber_covariance_scale_rejects_boolean_nis(invalid_nis):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        huber_covariance_scale(invalid_nis, huber_threshold=2.0)


@pytest.mark.parametrize(
    "invalid_nis",
    [-1.0, array([0.0, -1.0]), float("nan"), float("inf"), -float("inf")],
)
def test_student_t_covariance_scale_rejects_invalid_nis(invalid_nis):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        student_t_covariance_scale(invalid_nis, measurement_dim=1)


@pytest.mark.parametrize(
    "invalid_nis",
    [True, np.bool_(True), array([True])],
)
def test_student_t_covariance_scale_rejects_boolean_nis(invalid_nis):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        student_t_covariance_scale(invalid_nis, measurement_dim=1)


@pytest.mark.parametrize(
    "invalid_threshold", [float("nan"), float("inf"), -float("inf")]
)
def test_huber_covariance_scale_rejects_nonfinite_threshold(invalid_threshold):
    with pytest.raises(ValueError, match="finite and positive"):
        huber_covariance_scale(1.0, huber_threshold=invalid_threshold)


@pytest.mark.parametrize("invalid_dof", [float("nan"), float("inf"), -float("inf")])
def test_student_t_covariance_scale_rejects_nonfinite_dof(invalid_dof):
    with pytest.raises(ValueError, match="finite and greater than 2"):
        student_t_covariance_scale(1.0, measurement_dim=1, dof=invalid_dof)


@pytest.mark.parametrize(
    "invalid_min_scale",
    [float("nan"), float("inf"), -float("inf")],
)
def test_student_t_covariance_scale_rejects_nonfinite_min_scale(invalid_min_scale):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        student_t_covariance_scale(
            1.0,
            measurement_dim=1,
            min_scale=invalid_min_scale,
        )


def test_normalized_innovation_squared_rejects_boolean_inputs():
    with pytest.raises(ValueError, match="innovation"):
        normalized_innovation_squared(array([True]), array([[1.0]]))

    with pytest.raises(ValueError, match="innovation_covariance"):
        normalized_innovation_squared(array([0.0]), array([[True]]))


@pytest.mark.parametrize("invalid_scale", [float("nan"), float("inf"), -float("inf")])
def test_linear_gaussian_update_rejects_nonfinite_scale(invalid_scale):
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        linear_gaussian_update(
            array([0.0]),
            array([[1.0]]),
            array([0.0]),
            array([[1.0]]),
            array([[1.0]]),
            scale=invalid_scale,
        )


def test_linear_gaussian_update_rejects_boolean_array_inputs():
    base_kwargs = {
        "mean": array([0.0]),
        "covariance": array([[1.0]]),
        "measurement": array([0.0]),
        "measurement_matrix": array([[1.0]]),
        "meas_noise": array([[1.0]]),
    }
    invalid_overrides = (
        {"mean": array([True])},
        {"covariance": array([[True]])},
        {"measurement": array([True])},
        {"measurement_matrix": array([[True]])},
        {"meas_noise": array([[True]])},
    )

    for overrides in invalid_overrides:
        invalid_name = next(iter(overrides))
        kwargs = {**base_kwargs, **overrides}
        with pytest.raises(ValueError, match=invalid_name):
            linear_gaussian_update(**kwargs)
