"""Validated wrappers for motion-model catalog helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import motion_models as _motion_models

_nearly_coordinated_turn_model_impl = _motion_models.nearly_coordinated_turn_model
_continuous_to_discrete_lti_impl = _motion_models.continuous_to_discrete_lti


def _reject_complex_matrix(value: Any, name: str) -> None:
    """Reject complex-valued matrices before NumPy can discard imaginary parts."""
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError):
        return

    contains_complex_object = value_array.dtype == object and any(
        isinstance(item, (complex, np.complexfloating)) for item in value_array.flat
    )
    if np.iscomplexobj(value_array) or contains_complex_object:
        raise ValueError(f"{name} must contain real values")


def continuous_to_discrete_lti(
    continuous_matrix: Any,
    noise_input_matrix: Any | None = None,
    continuous_noise_covariance: Any | None = None,
    dt: float = 1.0,
) -> Any:
    """Discretize a real LTI model without silently truncating complex inputs."""
    _reject_complex_matrix(continuous_matrix, "continuous_matrix")
    if noise_input_matrix is not None:
        _reject_complex_matrix(noise_input_matrix, "noise_input_matrix")
    if continuous_noise_covariance is not None:
        _reject_complex_matrix(
            continuous_noise_covariance,
            "continuous_noise_covariance",
        )
    return _continuous_to_discrete_lti_impl(
        continuous_matrix,
        noise_input_matrix,
        continuous_noise_covariance,
        dt=dt,
    )


def nearly_coordinated_turn_model(
    dt: float = 1.0,
    position_spectral_density: float = 1.0,
    turn_rate_variance: float = 1e-4,
) -> Any:
    """Return a coordinated-turn model with validated nearly-constant-turn covariance."""
    dt = _motion_models._as_nonnegative_float(  # pylint: disable=protected-access
        dt,
        "dt",
    )
    turn_rate_variance = (
        _motion_models._as_nonnegative_float(  # pylint: disable=protected-access
            turn_rate_variance,
            "turn_rate_variance",
        )
    )
    return _nearly_coordinated_turn_model_impl(
        dt=dt,
        position_spectral_density=position_spectral_density,
        turn_rate_variance=turn_rate_variance,
    )


_motion_models.continuous_to_discrete_lti = continuous_to_discrete_lti
_motion_models.nearly_coordinated_turn_model = nearly_coordinated_turn_model

__all__ = ["continuous_to_discrete_lti", "nearly_coordinated_turn_model"]
