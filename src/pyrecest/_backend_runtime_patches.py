"""Runtime backend contract patches that must run after backend support setup."""

from __future__ import annotations


def patch_pytorch_close_equal_nan_device_contract() -> None:
    """Preserve ``equal_nan`` while keeping PyTorch close operands on one device."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    helper_names = ("isclose", "allclose")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_close_equal_nan_device_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if active_pytorch_backend:
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    def _preferred_device(*values):
        for value in values:
            if torch.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch.is_tensor(value):
                return value.device
        return None

    def _tensor_on_device(value, *, device):
        if torch.is_tensor(value):
            if device is not None and value.device != device:
                return value.to(device=device)
            return value
        return torch.as_tensor(value, device=device)

    def _comparison_operands(a, b):
        device = _preferred_device(a, b)
        a = _tensor_on_device(a, device=device)
        b = _tensor_on_device(b, device=device)
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        return torch.broadcast_tensors(a, b)

    def isclose(a, b, rtol=raw_pytorch.rtol, atol=raw_pytorch.atol, equal_nan=False):
        a, b = _comparison_operands(a, b)
        return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def allclose(a, b, atol=raw_pytorch.atol, rtol=raw_pytorch.rtol, equal_nan=False):
        a, b = _comparison_operands(a, b)
        return torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)

    for helper_name, helper in {
        "isclose": isclose,
        "allclose": allclose,
    }.items():
        previous = getattr(raw_pytorch, helper_name, None)
        helper.__name__ = getattr(previous, "__name__", helper_name)
        helper.__doc__ = getattr(previous, "__doc__", None)
        helper._pyrecest_device_contract = True
        helper._pyrecest_missing_value_contract = True
        helper._pyrecest_close_equal_nan_device_contract = True
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def patch_pytorch_repeat_numpy_contract() -> None:
    """Preserve the raw PyTorch repeat contract for non-PyTorch public backends."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from pyrecest._backend_submodules import (  # pylint: disable=import-outside-toplevel
            _pytorch_repeat_with_arraylike_inputs,
        )
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_repeat = getattr(raw_pytorch, "repeat", None)
    if original_repeat is None:
        return
    if getattr(original_repeat, "_pyrecest_repeat_contract", False):
        if active_pytorch_backend:
            setattr(backend, "repeat", original_repeat)
        return

    repeat = _pytorch_repeat_with_arraylike_inputs(
        original_repeat,
        raw_pytorch.array,
        np,
        torch,
    )
    raw_pytorch.repeat = repeat
    if active_pytorch_backend:
        backend.repeat = repeat


def patch_pytorch_dot_numpy_contract() -> None:
    """Make PyTorch ``dot`` follow NumPy's matrix/vector contraction contract."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_dot = getattr(raw_pytorch, "dot", None)
    if getattr(original_dot, "_pyrecest_numpy_dot_contract", False):
        if active_pytorch_backend:
            backend.dot = original_dot
        return

    def _preferred_device(*values):
        for value in values:
            if torch.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch.is_tensor(value):
                return value.device
        return None

    def _tensor_on_device(value, *, device):
        if torch.is_tensor(value):
            if device is not None and value.device != device:
                return value.to(device=device)
            return value
        return torch.as_tensor(value, device=device)

    def _promoted_pair(a, b):
        device = _preferred_device(a, b)
        a = _tensor_on_device(a, device=device)
        b = _tensor_on_device(b, device=device)
        dtype = torch.promote_types(a.dtype, b.dtype)
        return a.to(dtype=dtype), b.to(dtype=dtype)

    def dot(a, b):
        a, b = _promoted_pair(a, b)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        if a.ndim == 1 and b.ndim == 1:
            return torch.dot(a, b)
        if b.ndim == 1:
            return torch.tensordot(a, b, dims=([-1], [0]))
        if a.ndim == 1:
            return torch.tensordot(a, b, dims=([0], [-2]))
        return torch.tensordot(a, b, dims=([-1], [-2]))

    dot.__name__ = getattr(original_dot, "__name__", "dot")
    dot.__doc__ = getattr(original_dot, "__doc__", None)
    dot._pyrecest_numpy_dot_contract = True
    raw_pytorch.dot = dot
    if active_pytorch_backend:
        backend.dot = dot
