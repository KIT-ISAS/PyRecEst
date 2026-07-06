"""PyTorch sort helpers for backend compatibility."""

from __future__ import annotations

from operator import index as _operator_index


def normalize_sort_axis(axis):
    """Return a sort axis while preserving NumPy's flatten-all sentinel."""
    if axis is None:
        return None
    return _operator_index(axis)


def resolve_sort_stability(kind, stable):
    """Return PyTorch's stable flag for NumPy-style sort keyword aliases."""
    if kind is None:
        return stable
    if kind in {"stable", "mergesort"}:
        if stable is False:
            raise TypeError("sort() got conflicting 'kind' and 'stable' arguments")
        return True
    if kind in {"quicksort", "heapsort"}:
        if stable is True:
            raise TypeError("sort() got conflicting 'kind' and 'stable' arguments")
        return False
    raise ValueError(
        "sort kind must be one of 'quicksort', 'heapsort', 'stable', or 'mergesort'"
    )


def sort_axis_none(
    backend_module,
    torch_module,
    values,
    axis=-1,
    kind=None,
    order=None,
    *,
    stable=None,
    descending=False,
):
    """Sort values with NumPy-style ``axis=None`` and PyTorch ordering support."""
    if order is not None:
        raise ValueError("order is not supported by this backend")
    stable = resolve_sort_stability(kind, stable)
    values = backend_module.array(values)
    axis = normalize_sort_axis(axis)
    if axis is None:
        values = torch_module.flatten(values)
        axis = 0
    return torch_module.sort(
        values,
        dim=axis,
        descending=descending,
        stable=bool(stable) if stable is not None else False,
    ).values
