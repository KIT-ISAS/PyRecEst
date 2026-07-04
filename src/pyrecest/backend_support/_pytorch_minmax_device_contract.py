"""PyTorch binary-helper and linspace device compatibility hooks."""

from __future__ import annotations

import importlib
from operator import index as _operator_index


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type == "meta":
            return value.device
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _binary_operands(raw_pytorch, torch_module, left, right):
    """Return operands on a common dtype and an existing preferred device."""
    device = _preferred_pytorch_device(torch_module, left, right)
    left = raw_pytorch.array(left)
    right = raw_pytorch.array(right)
    dtype = torch_module.promote_types(left.dtype, right.dtype)
    if device is None:
        return left.to(dtype=dtype), right.to(dtype=dtype)
    return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)


# Backwards-compatible alias for existing imports/tests that used the old helper name.
_minmax_operands = _binary_operands


def _linspace_endpoint(torch_module, value, *, dtype, device):
    """Return a linspace endpoint tensor without copying off a preferred device."""
    if not torch_module.is_tensor(value):
        return torch_module.as_tensor(value, dtype=dtype, device=device)
    if dtype is None and (device is None or value.device == device):
        return value
    return value.to(dtype=dtype if dtype is not None else value.dtype, device=device)


def _linspace_result_dtype(raw_pytorch, torch_module, dtype, start, stop):
    """Return the NumPy-compatible result dtype for PyTorch ``linspace``."""
    result_dtype = dtype if dtype is not None else torch_module.result_type(start, stop)
    if dtype is None and not (result_dtype.is_floating_point or result_dtype.is_complex):
        result_dtype = raw_pytorch.get_default_dtype()
    return result_dtype


def _linspace_fraction_dtype(raw_pytorch, torch_module, result_dtype):
    """Return the floating dtype used to construct normalized fractions."""
    if result_dtype == torch_module.complex64:
        return torch_module.float32
    if result_dtype == torch_module.complex128:
        return torch_module.float64
    if not result_dtype.is_floating_point:
        return raw_pytorch.get_default_dtype()
    return result_dtype


def _linspace_integer_result_dtype(torch_module, result_dtype):
    """Return an explicit integer linspace dtype, if one was requested."""
    integer_dtypes = {
        torch_module.uint8,
        torch_module.int8,
        torch_module.int16,
        torch_module.int32,
        torch_module.int64,
    }
    return result_dtype if result_dtype in integer_dtypes else None


def _patch_pytorch_linspace_device_contract(raw_pytorch, backend, torch_module) -> None:
    """Patch raw/public PyTorch ``linspace`` to prefer existing non-CPU devices."""
    original_linspace = getattr(raw_pytorch, "linspace", None)
    if original_linspace is None:
        return
    if getattr(original_linspace, "_pyrecest_linspace_device_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.linspace = original_linspace
        return

    def linspace(start, stop, num=50, endpoint=True, dtype=None):
        num = _operator_index(num)
        if num < 0:
            raise ValueError("num must be non-negative")

        device = _preferred_pytorch_device(torch_module, start, stop)
        start_tensor = _linspace_endpoint(
            torch_module,
            start,
            dtype=None,
            device=device,
        )
        stop_tensor = _linspace_endpoint(
            torch_module,
            stop,
            dtype=None,
            device=device,
        )

        result_dtype = _linspace_result_dtype(
            raw_pytorch,
            torch_module,
            dtype,
            start_tensor,
            stop_tensor,
        )
        fraction_dtype = _linspace_fraction_dtype(
            raw_pytorch,
            torch_module,
            result_dtype,
        )
        integer_dtype = _linspace_integer_result_dtype(torch_module, result_dtype)
        computation_dtype = fraction_dtype if integer_dtype is not None else result_dtype
        start_tensor = start_tensor.to(dtype=computation_dtype)
        stop_tensor = stop_tensor.to(dtype=computation_dtype)
        start_tensor, stop_tensor = torch_module.broadcast_tensors(
            start_tensor,
            stop_tensor,
        )
        fractions = torch_module.arange(
            num,
            dtype=fraction_dtype,
            device=start_tensor.device,
        )
        denominator = num - 1 if endpoint and num > 1 else num
        if denominator > 0:
            fractions = fractions / denominator
        fractions = fractions.reshape((num,) + (1,) * start_tensor.ndim)

        result = start_tensor + (stop_tensor - start_tensor) * fractions
        if integer_dtype is not None:
            result = torch_module.floor(result)
        if result.dtype != result_dtype:
            result = result.to(dtype=result_dtype)
        return result

    linspace.__name__ = getattr(original_linspace, "__name__", "linspace")
    linspace.__doc__ = getattr(original_linspace, "__doc__", None)
    linspace._pyrecest_linspace_device_contract = True
    linspace._pyrecest_device_contract = True
    raw_pytorch.linspace = linspace
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.linspace = linspace


def _raw_pytorch_module():
    """Return the raw PyTorch backend module, importing it when available."""
    try:
        return importlib.import_module("pyrecest._backend.pytorch")
    except ModuleNotFoundError:
        return None


def _patch_binary_helpers(raw_pytorch, backend, torch_module, helpers, contract_attr):
    """Patch raw/public binary helpers to share dtype and preserve device."""
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            contract_attr,
            False,
        )
        for helper_name in helpers
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helpers:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    for helper_name, torch_helper in helpers.items():
        original_helper = getattr(raw_pytorch, helper_name)

        def binary_helper(left, right, _torch_helper=torch_helper):
            left, right = _binary_operands(raw_pytorch, torch_module, left, right)
            return _torch_helper(left, right)

        binary_helper.__name__ = getattr(original_helper, "__name__", helper_name)
        binary_helper.__doc__ = getattr(original_helper, "__doc__", None)
        setattr(binary_helper, contract_attr, True)
        binary_helper._pyrecest_device_contract = True
        setattr(raw_pytorch, helper_name, binary_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, binary_helper)


def patch_pytorch_minmax_device_contract() -> None:
    """Patch raw/public PyTorch binary helpers to preserve device placement."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch = _raw_pytorch_module()
    if raw_pytorch is None:  # pragma: no cover - backend import failed earlier
        return

    _patch_binary_helpers(
        raw_pytorch,
        backend,
        torch,
        {
            "maximum": torch.maximum,
            "minimum": torch.minimum,
        },
        "_pyrecest_minmax_device_contract",
    )
    _patch_binary_helpers(
        raw_pytorch,
        backend,
        torch,
        {
            "equal": torch.eq,
            "less_equal": torch.le,
        },
        "_pyrecest_comparison_device_contract",
    )
    _patch_pytorch_linspace_device_contract(raw_pytorch, backend, torch)
