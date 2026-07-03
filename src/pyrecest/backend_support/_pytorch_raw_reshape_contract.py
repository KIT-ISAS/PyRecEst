"""PyTorch raw ``reshape`` compatibility hook."""

from __future__ import annotations

from operator import index as _operator_index


def _pytorch_reshape_shape(shape, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style reshape dimensions for ``torch.reshape``."""
    if torch_module.is_tensor(shape):
        if shape.ndim == 0:
            return (_operator_index(shape.item()),)
        shape = shape.detach().cpu().tolist()
    elif getattr(shape, "ndim", None) == 0 and hasattr(shape, "item"):
        return (_operator_index(shape.item()),)

    try:
        return (_operator_index(shape),)
    except TypeError:
        pass

    if isinstance(shape, (str, bytes)):
        raise TypeError("reshape shape must be an integer or a sequence of integers")

    try:
        return tuple(_operator_index(dimension) for dimension in shape)
    except TypeError as exc:
        raise TypeError(
            "reshape shape must be an integer or a sequence of integers"
        ) from exc


def patch_pytorch_raw_reshape_contract() -> None:
    """Patch raw/public PyTorch ``reshape`` to accept NumPy-style inputs."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_reshape = getattr(raw_pytorch, "reshape", None)
    if original_reshape is None:
        return
    if getattr(original_reshape, "_pyrecest_raw_reshape_contract", False) or getattr(
        original_reshape,
        "_pyrecest_reshape_contract",
        False,
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.reshape = original_reshape
        return

    def reshape(x, shape):
        return original_reshape(
            raw_pytorch.array(x),
            _pytorch_reshape_shape(shape, torch_module),
        )

    reshape.__name__ = getattr(original_reshape, "__name__", "reshape")
    reshape.__doc__ = getattr(original_reshape, "__doc__", None)
    reshape._pyrecest_raw_reshape_contract = True
    raw_pytorch.reshape = reshape
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.reshape = reshape


__all__ = ["patch_pytorch_raw_reshape_contract"]
