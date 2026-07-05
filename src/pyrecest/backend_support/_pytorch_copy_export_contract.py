"""Compatibility hook for the top-level PyTorch ``copy`` export."""

from __future__ import annotations

import sys


def patch_pytorch_copy_export_contract() -> None:
    """Keep raw, public-backend, and package-level PyTorch ``copy`` synchronized."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    original_copy = getattr(raw_pytorch, "copy", None)
    if original_copy is None:
        return

    if getattr(original_copy, "_pyrecest_numpy_contract", False):
        copy = original_copy
    else:

        def copy(x):
            if raw_pytorch.is_array(x):
                return original_copy(x)
            return raw_pytorch.array(x)

        copy.__name__ = getattr(original_copy, "__name__", "copy")
        copy.__doc__ = getattr(original_copy, "__doc__", None)
        copy._pyrecest_numpy_contract = True
        raw_pytorch.copy = copy

    backend.copy = copy
    pyrecest_module = sys.modules.get("pyrecest")
    if pyrecest_module is not None:
        pyrecest_module.copy = copy


__all__ = ["patch_pytorch_copy_export_contract"]
