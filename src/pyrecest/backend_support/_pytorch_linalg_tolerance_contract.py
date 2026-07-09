"""Runtime patch for PyTorch linalg tolerance validation."""

from __future__ import annotations

from functools import wraps

import numpy as np

_TOLERANCE_TYPE_MESSAGE = (
    "linalg tolerances must be real numeric, not boolean or temporal"
)
_BOOLEAN_TYPES = (bool, np.bool_)
_TEMPORAL_SCALAR_TYPES = (np.datetime64, np.timedelta64)
_TEMPORAL_DTYPE_KINDS = {"M", "m"}


def _contains_boolean_value(value, torch_module) -> bool:
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    if torch_module.is_tensor(value):
        return value.dtype == torch_module.bool
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(
                _contains_boolean_value(item, torch_module)
                for item in value.reshape(-1)
            )
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_value(item, torch_module) for item in value)
    return False


def _contains_temporal_value(value) -> bool:
    if isinstance(value, _TEMPORAL_SCALAR_TYPES):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind in _TEMPORAL_DTYPE_KINDS:
            return True
        if value.dtype == object:
            return any(_contains_temporal_value(item) for item in value.reshape(-1))
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_temporal_value(item) for item in value)
    return False


def _validate_linalg_tolerance_argument(value, torch_module) -> None:
    if value is None:
        return
    if _contains_boolean_value(value, torch_module) or _contains_temporal_value(value):
        raise TypeError(_TOLERANCE_TYPE_MESSAGE)


def _wrap_pinv(pinv, torch_module):
    if getattr(pinv, "_pyrecest_linalg_tolerance_contract", False):
        return pinv

    @wraps(pinv)
    def pinv_with_tolerance_validation(
        a, rcond=None, hermitian=False, *, atol=None, rtol=None, out=None
    ):
        _validate_linalg_tolerance_argument(rcond, torch_module)
        _validate_linalg_tolerance_argument(atol, torch_module)
        _validate_linalg_tolerance_argument(rtol, torch_module)
        return pinv(a, rcond=rcond, hermitian=hermitian, atol=atol, rtol=rtol, out=out)

    pinv_with_tolerance_validation._pyrecest_linalg_tolerance_contract = True
    return pinv_with_tolerance_validation


def _wrap_matrix_rank(matrix_rank, torch_module):
    if getattr(matrix_rank, "_pyrecest_linalg_tolerance_contract", False):
        return matrix_rank

    @wraps(matrix_rank)
    def matrix_rank_with_tolerance_validation(
        a, tol=None, hermitian=False, *, rtol=None, atol=None, **kwargs
    ):
        _validate_linalg_tolerance_argument(tol, torch_module)
        _validate_linalg_tolerance_argument(atol, torch_module)
        _validate_linalg_tolerance_argument(rtol, torch_module)
        return matrix_rank(
            a,
            tol=tol,
            hermitian=hermitian,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    matrix_rank_with_tolerance_validation._pyrecest_linalg_tolerance_contract = True
    return matrix_rank_with_tolerance_validation


def _patch_linalg_module(linalg_module, torch_module) -> None:
    pinv = getattr(linalg_module, "pinv", None)
    if pinv is not None:
        linalg_module.pinv = _wrap_pinv(pinv, torch_module)

    matrix_rank = getattr(linalg_module, "matrix_rank", None)
    if matrix_rank is not None:
        linalg_module.matrix_rank = _wrap_matrix_rank(matrix_rank, torch_module)


def patch_pytorch_linalg_tolerance_contract() -> None:
    """Make PyTorch linalg helpers reject boolean and temporal tolerances."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_linalg = getattr(raw_pytorch, "linalg", None)
    if raw_linalg is not None:
        _patch_linalg_module(raw_linalg, torch)

    backend_linalg = getattr(backend, "linalg", None)
    if (
        backend_linalg is not None
        and getattr(backend, "__backend_name__", None) == "pytorch"
        and backend_linalg is not raw_linalg
    ):
        _patch_linalg_module(backend_linalg, torch)
