"""Numpy based random backend."""

from operator import index as _operator_index

import numpy as _np
from numpy.random import default_rng as _default_rng
from numpy.random import (  # For PyRecEst
    get_state,
    seed,
    set_state,
)

from .._shared_numpy.random import (
    _normalize_probability_values,
    _normalize_size,
    choice as _shared_choice,
    multivariate_normal,
    normal,
    rand,
    uniform,
)

_BOOLEAN_TYPES = (bool, _np.bool_)
_CHOICE_POPULATION_ERROR = "a must be a positive integer or a non-empty array"


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _validate_randint_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must contain integer values")
    try:
        bound_array = _np.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain integer values") from exc
    if bound_array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer values")


def _validate_randint_dtype(dtype):
    try:
        dtype = _np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError("dtype must be an integer dtype") from exc
    if dtype.kind not in "iu":
        raise TypeError("dtype must be an integer dtype")
    return dtype


def randint(low, high=None, size=None, dtype=int):
    """Draw integer samples after rejecting non-integer bounds and dtypes."""

    size = _normalize_size(size)
    dtype = _validate_randint_dtype(dtype)
    if high is None:
        _validate_randint_bound(low, "high")
        return _np.random.randint(low, high=None, size=size, dtype=dtype)

    _validate_randint_bound(low, "low")
    _validate_randint_bound(high, "high")
    return _np.random.randint(low, high=high, size=size, dtype=dtype)


def _normalize_choice_axis(axis, ndim):
    if isinstance(axis, _BOOLEAN_TYPES):
        raise TypeError("axis must be an integer")
    try:
        axis = _operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _validate_choice_population_size(a, axis):
    a_array = _np.asarray(a)
    if a_array.ndim == 0:
        scalar = a_array.item()
        if isinstance(scalar, _BOOLEAN_TYPES):
            raise ValueError(_CHOICE_POPULATION_ERROR)
        try:
            population_size = _operator_index(scalar)
        except TypeError:
            return
        if population_size <= 0:
            raise ValueError(_CHOICE_POPULATION_ERROR)
        return

    axis = _normalize_choice_axis(axis, a_array.ndim)
    if a_array.shape[axis] <= 0:
        raise ValueError(_CHOICE_POPULATION_ERROR)


def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    """Draw samples from a non-empty integer or array population."""

    _validate_choice_population_size(a, axis)
    return _shared_choice(a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)


def _validate_multinomial_sample_count(n):
    if _contains_boolean_value(n):
        raise TypeError("n must be a non-negative integer")
    try:
        n_array = _np.asarray(n)
    except (TypeError, ValueError) as exc:
        raise TypeError("n must be a non-negative integer") from exc
    if n_array.shape != () or n_array.dtype.kind not in "iu":
        raise TypeError("n must be a non-negative integer")
    count = int(n_array.item())
    if count < 0:
        raise ValueError("n must be non-negative")
    return count


def _validate_multinomial_pvals(pvals):
    if _contains_boolean_value(pvals):
        raise TypeError("pvals must be real numeric, not boolean")
    try:
        pvals_array = _np.asarray(pvals)
    except (TypeError, ValueError) as exc:
        raise TypeError("pvals must be real numeric") from exc
    if pvals_array.dtype.kind not in "iuf":
        raise TypeError("pvals must be real numeric")

    pvals_array = pvals_array.astype(float, copy=False)
    if pvals_array.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if pvals_array.size == 0:
        raise ValueError("pvals must contain at least one probability")

    return _normalize_probability_values(pvals_array)


def _validate_multinomial_size(size):
    return _normalize_size(size)


def multinomial(n, pvals, size=None):
    n = _validate_multinomial_sample_count(n)
    pvals_array = _validate_multinomial_pvals(pvals)
    size = _validate_multinomial_size(size)
    return _np.random.multinomial(n, pvals_array, size=size)
