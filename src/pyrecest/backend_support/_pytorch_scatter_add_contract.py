"""Compatibility hook for PyTorch ``scatter_add`` backend semantics."""

from __future__ import annotations

from operator import index as _operator_index


def _normalize_pytorch_scatter_dim(dim, torch_module) -> int:
    """Return one PyTorch scatter dimension while rejecting booleans."""
    if isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    if torch_module.is_tensor(dim):
        if dim.ndim != 0 or dim.dtype == torch_module.bool:
            raise TypeError("dim must be an integer")
        dim = dim.item()
    try:
        return _operator_index(dim)
    except TypeError as exc:
        raise TypeError("dim must be an integer") from exc


def patch_pytorch_scatter_add_contract() -> None:
    """Make raw/public PyTorch ``scatter_add`` accept array-like operands."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_scatter_add = getattr(raw_pytorch, "scatter_add", None)
    if original_scatter_add is None:
        return
    if getattr(original_scatter_add, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.scatter_add = original_scatter_add
        return

    def scatter_add(input, dim, index, src):  # pylint: disable=redefined-builtin
        values = raw_pytorch.array(input)
        dim_value = _normalize_pytorch_scatter_dim(dim, torch)
        if not torch.is_tensor(index):
            index = torch.as_tensor(index, dtype=torch.long, device=values.device)
        else:
            index = index.to(device=values.device, dtype=torch.long)
        if not torch.is_tensor(src):
            src = torch.as_tensor(src, dtype=values.dtype, device=values.device)
        else:
            src = src.to(device=values.device, dtype=values.dtype)
        return torch.scatter_add(values, dim_value, index, src)

    scatter_add.__name__ = getattr(original_scatter_add, "__name__", "scatter_add")
    scatter_add.__doc__ = getattr(original_scatter_add, "__doc__", None)
    scatter_add._pyrecest_arraylike_contract = True
    scatter_add._pyrecest_numpy_contract = True
    raw_pytorch.scatter_add = scatter_add
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.scatter_add = scatter_add


__all__ = ["patch_pytorch_scatter_add_contract"]
