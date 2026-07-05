"""PyTorch ``one_hot`` scalar-label compatibility hook."""

from __future__ import annotations

from operator import index as _operator_index


def _is_numpy_bool_scalar(value) -> bool:
    return type(value).__module__ == "numpy" and type(value).__name__ == "bool_"


def _contains_boolean_label(value, torch_module) -> bool:
    """Return whether labels contain booleans before coercing to ``long``."""
    if isinstance(value, bool) or _is_numpy_bool_scalar(value):
        return True
    if torch_module.is_tensor(value):
        return value.dtype == torch_module.bool
    dtype = getattr(value, "dtype", None)
    if dtype is not None and str(dtype).lower() in {"bool", "bool_"}:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_label(item, torch_module) for item in value)
    return False


def patch_pytorch_one_hot_scalar_contract() -> None:
    """Patch raw/public PyTorch ``one_hot`` to handle scalar labels correctly."""
    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_one_hot = getattr(pytorch_backend, "one_hot", None)
    if original_one_hot is None:
        return
    if getattr(original_one_hot, "_pyrecest_scalar_label_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.one_hot = original_one_hot
        return

    def one_hot(labels, num_classes):
        num_classes = _operator_index(num_classes)
        if _contains_boolean_label(labels, torch_module):
            raise TypeError("one_hot labels must be integers, not booleans")
        if torch_module.is_tensor(labels):
            if labels.dtype.is_floating_point or labels.dtype.is_complex:
                return original_one_hot(labels, num_classes)
            labels = labels.to(dtype=torch_module.long)
        else:
            labels = torch_module.as_tensor(labels, dtype=torch_module.long)
        return torch_module.nn.functional.one_hot(labels, num_classes).to(
            dtype=torch_module.uint8
        )

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_scalar_label_contract = True
    pytorch_backend.one_hot = one_hot
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.one_hot = one_hot


__all__ = ["patch_pytorch_one_hot_scalar_contract"]
