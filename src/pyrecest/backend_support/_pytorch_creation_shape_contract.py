"""PyTorch creation-helper shape validation hook."""

from __future__ import annotations

from operator import index as _operator_index


def _pytorch_creation_dimension(dimension, numpy_module) -> int:
    """Return one NumPy-style shape dimension as a non-boolean integer."""
    if isinstance(dimension, (bool, numpy_module.bool_)):
        raise TypeError("shape dimensions must be integers")
    try:
        return _operator_index(dimension)
    except TypeError as exc:
        raise TypeError("shape dimensions must be integers") from exc


def _pytorch_creation_shape(shape, numpy_module, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style creation shapes while rejecting boolean dimensions."""
    if torch_module.is_tensor(shape):
        shape = shape.detach().cpu().numpy()
    if isinstance(shape, (list, tuple)):
        normalized_shape = tuple(
            _pytorch_creation_dimension(one_dimension, numpy_module)
            for one_dimension in shape
        )
        if any(one_dimension < 0 for one_dimension in normalized_shape):
            raise ValueError("negative dimensions are not allowed")
        return normalized_shape
    shape_array = numpy_module.asarray(shape)
    if shape_array.shape == ():
        normalized_shape = (
            _pytorch_creation_dimension(shape_array.item(), numpy_module),
        )
    else:
        normalized_shape = tuple(
            _pytorch_creation_dimension(one_dimension, numpy_module)
            for one_dimension in shape_array.tolist()
        )
    if any(one_dimension < 0 for one_dimension in normalized_shape):
        raise ValueError("negative dimensions are not allowed")
    return normalized_shape


def patch_pytorch_creation_shape_contract() -> None:
    """Patch raw/public PyTorch creation helpers to reject boolean shapes."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from pyrecest._backend.pytorch._common import (  # pylint: disable=import-outside-toplevel
            _normalize_dtype,
        )
        from pyrecest.backend_support._pytorch_scatter_add_contract import (  # pylint: disable=import-outside-toplevel
            patch_pytorch_scatter_add_contract,
        )
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    patch_pytorch_scatter_add_contract()

    def _wrap_creation_helper(helper_name, torch_helper, *, has_fill_value=False):
        original_helper = getattr(raw_pytorch, helper_name, None)
        if original_helper is None:
            return None
        if getattr(original_helper, "_pyrecest_boolean_shape_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, original_helper)
            return original_helper

        if has_fill_value:

            def creation_helper(shape, fill_value, dtype=None, *args, **kwargs):
                return torch_helper(
                    _pytorch_creation_shape(shape, np, torch),
                    fill_value,
                    *args,
                    dtype=_normalize_dtype(dtype),
                    **kwargs,
                )

        else:

            def creation_helper(shape, dtype=None, *args, **kwargs):
                return torch_helper(
                    _pytorch_creation_shape(shape, np, torch),
                    *args,
                    dtype=_normalize_dtype(dtype),
                    **kwargs,
                )

        creation_helper.__name__ = getattr(original_helper, "__name__", helper_name)
        creation_helper.__doc__ = getattr(original_helper, "__doc__", None)
        creation_helper._pyrecest_numpy_contract = True
        creation_helper._pyrecest_boolean_shape_contract = True
        return creation_helper

    for helper_name, torch_helper, has_fill_value in (
        ("empty", torch.empty, False),
        ("zeros", torch.zeros, False),
        ("ones", torch.ones, False),
        ("full", torch.full, True),
    ):
        helper = _wrap_creation_helper(
            helper_name,
            torch_helper,
            has_fill_value=has_fill_value,
        )
        if helper is None:
            continue
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)
