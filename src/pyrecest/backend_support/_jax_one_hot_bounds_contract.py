"""JAX ``one_hot`` bounds compatibility hook."""

from __future__ import annotations

from operator import index as _operator_index


def patch_jax_one_hot_bounds_contract() -> None:
    """Patch raw/public JAX ``one_hot`` to zero labels outside the class range."""
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as jax_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    original_one_hot = getattr(jax_backend, "one_hot", None)
    if original_one_hot is None:
        return
    if getattr(original_one_hot, "_pyrecest_label_bounds_contract", False):
        if getattr(backend, "__backend_name__", None) == "jax":
            backend.one_hot = original_one_hot
        return

    def _uint8_one_hot(labels, num_classes):
        num_classes = _operator_index(num_classes)
        labels = jnp.asarray(labels)
        if labels.dtype.kind not in "iu":
            raise TypeError("one_hot labels must be integer-valued")

        valid = (labels >= 0) & (labels < num_classes)
        safe_labels = jnp.where(valid, labels, 0)
        encoded = jnp.eye(num_classes, dtype=jnp.uint8)[safe_labels]
        return jnp.where(valid[..., None], encoded, jnp.zeros_like(encoded))

    def one_hot(labels=None, num_classes=None, *, indices=None, depth=None):
        if indices is not None:
            if labels is not None:
                raise TypeError("one_hot() got both 'labels' and 'indices'")
            labels = indices
        if depth is not None:
            if num_classes is not None and num_classes != depth:
                raise TypeError("one_hot() got both 'num_classes' and 'depth'")
            num_classes = depth
        if labels is None:
            raise TypeError("one_hot() missing required argument 'labels'")
        if num_classes is None:
            raise TypeError("one_hot() missing required argument 'num_classes'")
        return _uint8_one_hot(labels, num_classes)

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_backend_contract = True
    one_hot._pyrecest_label_bounds_contract = True
    jax_backend.one_hot = one_hot
    if getattr(backend, "__backend_name__", None) == "jax":
        backend.one_hot = one_hot


__all__ = ["patch_jax_one_hot_bounds_contract"]
