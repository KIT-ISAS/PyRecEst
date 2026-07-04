"""PyTorch ``allclose`` compatibility hook."""

from __future__ import annotations

from pyrecest.backend_support._pytorch_creation_shape_contract import (
    patch_pytorch_creation_shape_contract as _patch_pytorch_creation_shape_contract,
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


def _coerce_binary_args(torch_module, x, y):
    """Move array-like PyTorch binary operands to a preferred existing device."""

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


def patch_pytorch_allclose_device_contract() -> None:
    """Patch raw/public PyTorch ``allclose`` to preserve non-CPU operands."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_pytorch_creation_shape_contract()

    original_allclose = getattr(pytorch_backend, "allclose", None)
    if original_allclose is None:
        return
    if getattr(original_allclose, "_pyrecest_equal_nan_device_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.allclose = original_allclose
        return

    def allclose(
        a,
        b,
        atol=pytorch_backend.atol,
        rtol=pytorch_backend.rtol,
        equal_nan=False,
    ):
        a, b = _coerce_binary_args(torch_module, a, b)
        a, b = pytorch_backend.convert_to_wider_dtype([a, b])
        a, b = torch_module.broadcast_tensors(a, b)
        return torch_module.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    allclose.__name__ = getattr(original_allclose, "__name__", "allclose")
    allclose.__doc__ = getattr(original_allclose, "__doc__", None)
    allclose._pyrecest_equal_nan_device_contract = True
    pytorch_backend.allclose = allclose
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.allclose = allclose
