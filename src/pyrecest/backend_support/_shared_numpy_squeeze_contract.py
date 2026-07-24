"""Compatibility patch for NumPy-style squeeze axis validation."""

from __future__ import annotations


def patch_shared_numpy_squeeze_nonsingleton_axis_contract() -> None:
    """Reject explicitly selected axes whose extent is not one."""
    try:
        import pyrecest._backend._shared_numpy as shared_numpy  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - source tree corruption only
        return

    if getattr(backend, "__backend_name__", None) not in {"autograd", "numpy"}:
        return

    original_squeeze = shared_numpy.squeeze
    if getattr(original_squeeze, "_pyrecest_nonsingleton_axis_contract", False):
        backend.squeeze = original_squeeze
        return

    np_module = shared_numpy._np

    def squeeze(x, axis=None):
        values = np_module.asarray(x)
        if axis is not None:
            axes = shared_numpy._normalize_squeeze_axes(axis)
            for one_axis in axes:
                normalized_axis = one_axis + values.ndim if one_axis < 0 else one_axis
                if 0 <= normalized_axis < values.ndim and values.shape[normalized_axis] != 1:
                    raise ValueError(
                        "cannot select an axis to squeeze out which has size not equal to one"
                    )
        return original_squeeze(values, axis=axis)

    squeeze.__name__ = getattr(original_squeeze, "__name__", "squeeze")
    squeeze.__doc__ = getattr(original_squeeze, "__doc__", None)
    squeeze._pyrecest_nonsingleton_axis_contract = True
    shared_numpy.squeeze = squeeze
    backend.squeeze = squeeze
