"""Small helpers for inspecting the process-global PyRecEst backend."""

from __future__ import annotations

import os
import warnings
from operator import index as _operator_index


def _pytorch_modules():
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return None, None, None
    return backend, pytorch_backend, torch_module


def _active_pytorch_backend(backend) -> bool:
    return getattr(backend, "__backend_name__", None) == "pytorch"


def _pytorch_scalar_tensor_index(index, torch_module):
    """Return Python int indices for scalar integer tensors."""
    if not torch_module.is_tensor(index) or index.ndim != 0:
        return index
    if (
        index.dtype in {torch_module.bool, torch_module.uint8}
        or index.dtype.is_floating_point
        or index.dtype.is_complex
    ):
        return index
    return _operator_index(index)


def _wrap_pytorch_assignment_helper(original_assignment, torch_module):
    """Normalize scalar tensor indices before assignment helper len() checks."""
    if getattr(original_assignment, "_pyrecest_scalar_tensor_index_contract", False):
        return original_assignment

    def assignment(x, values, indices, axis=0):
        indices = _pytorch_scalar_tensor_index(indices, torch_module)
        return original_assignment(x, values, indices, axis=axis)

    assignment.__name__ = getattr(original_assignment, "__name__", "assignment")
    assignment.__doc__ = getattr(original_assignment, "__doc__", None)
    assignment._pyrecest_scalar_tensor_index_contract = True
    return assignment


def _patch_pytorch_assignment_scalar_tensor_indices() -> None:
    """Make PyTorch assignment helpers accept scalar integer tensor indices."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None or not _active_pytorch_backend(backend):
        return
    backend.assignment = _wrap_pytorch_assignment_helper(backend.assignment, torch_module)
    backend.assignment_by_sum = _wrap_pytorch_assignment_helper(
        backend.assignment_by_sum, torch_module
    )
    pytorch_backend.assignment = _wrap_pytorch_assignment_helper(
        pytorch_backend.assignment, torch_module
    )
    pytorch_backend.assignment_by_sum = _wrap_pytorch_assignment_helper(
        pytorch_backend.assignment_by_sum, torch_module
    )


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _patch_pytorch_diag_numpy_contract() -> None:
    """Make PyTorch diag accept array-like inputs and NumPy's ``k`` keyword."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None or not _active_pytorch_backend(backend):
        return
    if getattr(pytorch_backend.diag, "_pyrecest_numpy_contract", False):
        backend.diag = pytorch_backend.diag
        return

    def diag(v, k=0):
        return torch_module.diag(pytorch_backend.array(v), diagonal=k)

    diag.__name__ = getattr(torch_module.diag, "__name__", "diag")
    diag.__doc__ = getattr(torch_module.diag, "__doc__", None)
    diag._pyrecest_numpy_contract = True
    backend.diag = diag
    pytorch_backend.diag = diag


def _patch_pytorch_broadcast_arrays_numpy_contract() -> None:
    """Make PyTorch broadcast_arrays accept NumPy-style array-like inputs."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None:
        return
    active = _active_pytorch_backend(backend)
    if getattr(pytorch_backend.broadcast_arrays, "_pyrecest_numpy_contract", False):
        if active:
            backend.broadcast_arrays = pytorch_backend.broadcast_arrays
        return

    def broadcast_arrays(*arrays):
        tensors = tuple(pytorch_backend.array(array) for array in arrays)
        return torch_module.broadcast_tensors(*tensors)

    broadcast_arrays.__name__ = "broadcast_arrays"
    broadcast_arrays.__doc__ = getattr(torch_module.broadcast_tensors, "__doc__", None)
    broadcast_arrays._pyrecest_numpy_contract = True
    pytorch_backend.broadcast_arrays = broadcast_arrays
    if active:
        backend.broadcast_arrays = broadcast_arrays


def _patch_pytorch_round_numpy_contract() -> None:
    """Make PyTorch round accept NumPy-style array-like inputs."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None or not _active_pytorch_backend(backend):
        return
    if getattr(pytorch_backend.round, "_pyrecest_numpy_contract", False):
        backend.round = pytorch_backend.round
        return

    def round(a, decimals=0, out=None):  # pylint: disable=redefined-builtin
        decimals = _operator_index(decimals)
        result = torch_module.round(pytorch_backend.array(a), decimals=decimals)
        if out is not None:
            out.copy_(result)
            return out
        return result

    round.__name__ = getattr(torch_module.round, "__name__", "round")
    round.__doc__ = getattr(torch_module.round, "__doc__", None)
    round._pyrecest_numpy_contract = True
    backend.round = round
    pytorch_backend.round = round


def _patch_pytorch_special_numpy_contract() -> None:
    """Make PyTorch special functions accept NumPy-style array-like inputs."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None:
        return
    active = _active_pytorch_backend(backend)

    def _return_or_store_out(result, out):
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _special_input(a):
        values = pytorch_backend.array(a)
        if not pytorch_backend.is_floating(values) and not pytorch_backend.is_complex(
            values
        ):
            values = pytorch_backend.cast(
                values, dtype=pytorch_backend.get_default_dtype()
            )
        return values

    def _gamma_from_lgamma(values):
        result = torch_module.exp(torch_module.special.gammaln(values))
        if pytorch_backend.is_complex(values):
            return result
        sign = torch_module.ones_like(result)
        negative = values < 0
        negative_zero = (values == 0) & torch_module.signbit(values)
        nonpositive_integer_pole = negative & (values == torch_module.floor(values))
        reflected_sign = torch_module.sign(torch_module.sin(torch_module.pi * values)).to(
            dtype=result.dtype
        )
        sign = torch_module.where(negative, reflected_sign, sign)
        sign = torch_module.where(negative_zero, -torch_module.ones_like(sign), sign)
        result = result * sign
        return torch_module.where(
            nonpositive_integer_pole,
            torch_module.full_like(result, float("nan")),
            result,
        )

    def erf(a, out=None):
        return _return_or_store_out(torch_module.erf(_special_input(a)), out)

    def gammaln(a, out=None):
        return _return_or_store_out(torch_module.special.gammaln(_special_input(a)), out)

    def gamma(a, out=None):
        return _return_or_store_out(_gamma_from_lgamma(_special_input(a)), out)

    def polygamma(n, a, out=None):
        return _return_or_store_out(torch_module.polygamma(n, _special_input(a)), out)

    for name, helper, target in (
        ("erf", erf, torch_module.erf),
        ("gammaln", gammaln, torch_module.special.gammaln),
        ("gamma", gamma, pytorch_backend.gamma),
        ("polygamma", polygamma, torch_module.polygamma),
    ):
        helper.__name__ = name
        helper.__doc__ = getattr(target, "__doc__", None)
        helper._pyrecest_numpy_contract = True
        setattr(pytorch_backend, name, helper)
        if active:
            setattr(backend, name, helper)
    pytorch_backend._gammaln = gammaln  # pylint: disable=protected-access


def _patch_pytorch_stack_helpers_numpy_contract() -> None:
    """Make PyTorch stack helpers accept NumPy-style array-like inputs."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None:
        return
    active = _active_pytorch_backend(backend)
    try:
        import numpy as numpy_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - NumPy is a core dependency
        return
    if getattr(pytorch_backend.hstack, "_pyrecest_numpy_contract", False):
        if active:
            for helper_name in ("hstack", "vstack", "column_stack", "dstack"):
                setattr(backend, helper_name, getattr(pytorch_backend, helper_name))
        return

    def _tensor_sequence(tup):
        return [pytorch_backend.array(item) for item in tup]

    def hstack(tup):
        tensors = [torch_module.atleast_1d(tensor) for tensor in _tensor_sequence(tup)]
        if not tensors:
            return torch_module.cat(tensors, dim=0)
        return torch_module.cat(tensors, dim=0 if tensors[0].ndim == 1 else 1)

    def vstack(tup):
        return torch_module.cat(
            [torch_module.atleast_2d(tensor) for tensor in _tensor_sequence(tup)], dim=0
        )

    def column_stack(tup):
        tensors = []
        for tensor in _tensor_sequence(tup):
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            tensors.append(tensor)
        return torch_module.cat(tensors, dim=1)

    def dstack(tup):
        return torch_module.cat(
            [torch_module.atleast_3d(tensor) for tensor in _tensor_sequence(tup)], dim=2
        )

    for helper_name, helper in {
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }.items():
        helper.__name__ = helper_name
        helper.__doc__ = getattr(numpy_module, helper_name).__doc__
        helper._pyrecest_numpy_contract = True
        setattr(pytorch_backend, helper_name, helper)
        if active:
            setattr(backend, helper_name, helper)


def _patch_pytorch_conj_numpy_contract() -> None:
    """Make PyTorch conj accept NumPy-style array-like inputs."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None:
        return
    active = _active_pytorch_backend(backend)
    if getattr(pytorch_backend.conj, "_pyrecest_numpy_contract", False):
        if active:
            backend.conj = pytorch_backend.conj
        return

    def conj(x):
        return torch_module.conj(pytorch_backend.array(x))

    conj.__name__ = getattr(pytorch_backend.conj, "__name__", "conj")
    conj.__doc__ = getattr(pytorch_backend.conj, "__doc__", None)
    conj._pyrecest_numpy_contract = True
    pytorch_backend.conj = conj
    if active:
        backend.conj = conj


def _patch_raw_pytorch_comparison_numpy_contract() -> None:
    """Make raw PyTorch comparison helpers accept NumPy-style array-like inputs."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None:
        return

    def _coerce_binary_args(x, y):
        device = _preferred_pytorch_device(torch_module, x, y)
        if not torch_module.is_tensor(x):
            x = torch_module.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not torch_module.is_tensor(y):
            y = torch_module.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        return x, y

    def _wrap_comparison(helper_name, torch_func):
        original = getattr(pytorch_backend, helper_name, None)
        if getattr(original, "_pyrecest_numpy_contract", False):
            return original

        def comparison(x, y, **kwargs):
            x, y = _coerce_binary_args(x, y)
            return torch_func(x, y, **kwargs)

        comparison.__name__ = getattr(torch_func, "__name__", helper_name)
        comparison.__doc__ = getattr(torch_func, "__doc__", None)
        comparison._pyrecest_numpy_contract = True
        return comparison

    for helper_name, torch_func in (
        ("greater", torch_module.greater),
        ("less", torch_module.less),
        ("logical_or", torch_module.logical_or),
        ("logical_and", torch_module.logical_and),
    ):
        helper = _wrap_comparison(helper_name, torch_func)
        setattr(pytorch_backend, helper_name, helper)
        if _active_pytorch_backend(backend):
            setattr(backend, helper_name, helper)


def _patch_raw_pytorch_isclose_equal_nan_contract() -> None:
    """Make PyTorch isclose accept NumPy's ``equal_nan`` keyword."""
    backend, pytorch_backend, torch_module = _pytorch_modules()
    if backend is None:
        return
    if getattr(pytorch_backend.isclose, "_pyrecest_equal_nan_contract", False):
        if _active_pytorch_backend(backend):
            backend.isclose = pytorch_backend.isclose
        return

    def isclose(x, y, rtol=pytorch_backend.rtol, atol=pytorch_backend.atol, equal_nan=False):
        device = _preferred_pytorch_device(torch_module, x, y)
        if not torch_module.is_tensor(x):
            x = torch_module.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not torch_module.is_tensor(y):
            y = torch_module.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        x, y = pytorch_backend.convert_to_wider_dtype([x, y])
        return torch_module.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    isclose.__name__ = getattr(pytorch_backend.isclose, "__name__", "isclose")
    isclose.__doc__ = getattr(pytorch_backend.isclose, "__doc__", None)
    isclose._pyrecest_equal_nan_contract = True
    pytorch_backend.isclose = isclose
    if _active_pytorch_backend(backend):
        backend.isclose = isclose


_patch_pytorch_assignment_scalar_tensor_indices()
_patch_pytorch_diag_numpy_contract()
_patch_pytorch_broadcast_arrays_numpy_contract()
_patch_pytorch_round_numpy_contract()
_patch_pytorch_special_numpy_contract()
_patch_pytorch_stack_helpers_numpy_contract()
_patch_pytorch_conj_numpy_contract()
_patch_raw_pytorch_comparison_numpy_contract()
_patch_raw_pytorch_isclose_equal_nan_contract()


def get_backend_name() -> str:
    """Return the backend selected at import time."""
    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    return backend.__backend_name__  # pylint: disable=no-member


def is_backend(expected: str | tuple[str, ...]) -> bool:
    """Return whether the active backend matches one of the expected names."""
    expected_names = _normalize_expected_backend_names(expected)
    return get_backend_name() in expected_names


def _normalize_expected_backend_names(
    expected: str | tuple[str, ...],
) -> tuple[str, ...]:
    message = "expected must name at least one backend."
    if isinstance(expected, str):
        names = (expected,)
    else:
        try:
            names = tuple(expected)
        except TypeError as exc:
            raise ValueError(message) from exc
    if not names or any(
        not isinstance(name, str) or not name or name.strip() != name for name in names
    ):
        raise ValueError(message)
    return tuple(dict.fromkeys(names))


def assert_backend(expected: str | tuple[str, ...]) -> None:
    """Raise ``RuntimeError`` unless the active backend matches ``expected``.

    Parameters
    ----------
    expected : str or tuple[str, ...]
        Allowed backend name or names.
    """
    active = get_backend_name()
    expected_names = _normalize_expected_backend_names(expected)
    if active not in expected_names:
        allowed = ", ".join(expected_names)
        raise RuntimeError(
            f"Expected PyRecEst backend {allowed}; active backend is {active}."
        )


def warn_if_backend_env_changed() -> None:
    """Warn when ``PYRECEST_BACKEND`` no longer matches the imported backend.

    Backend selection is process-global and import-time only. Changing the
    environment variable after importing :mod:`pyrecest` does not switch the
    already constructed backend facade.
    """
    active = get_backend_name()
    requested = os.environ.get("PYRECEST_BACKEND", active)
    if requested != active:
        warnings.warn(
            "PYRECEST_BACKEND was changed after pyrecest was imported. "
            f"The active backend remains {active!r}; the environment now requests "
            f"{requested!r}. Start a new Python process to switch backends.",
            RuntimeWarning,
            stacklevel=2,
        )
