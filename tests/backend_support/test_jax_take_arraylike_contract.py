import importlib.util

import pytest

pytestmark = pytest.mark.backend_portable


def _require_raw_jax_backend():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")
    import pyrecest  # noqa: F401  # triggers runtime backend patches
    import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel

    return raw_jax


def test_raw_jax_take_accepts_array_like_values_after_import():
    raw_jax = _require_raw_jax_backend()

    axis_result = raw_jax.take([[10, 20], [30, 40]], [1, 0], axis=1)
    assert raw_jax.to_numpy(axis_result).tolist() == [[20, 10], [40, 30]]

    flat_result = raw_jax.take([[10, 20], [30, 40]], [3, 0], axis=None)
    assert raw_jax.to_numpy(flat_result).tolist() == [40, 10]

    out_result = raw_jax.take(
        [[10, 20], [30, 40]],
        [1, 0],
        axis=1,
        out=raw_jax.zeros((2, 2)),
    )
    assert raw_jax.to_numpy(out_result).tolist() == [[20, 10], [40, 30]]


def test_public_jax_take_accepts_array_like_values_when_active():
    _require_raw_jax_backend()
    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "jax":
        pytest.skip("public JAX backend is not active")

    axis_result = backend.take([[10, 20], [30, 40]], [1, 0], axis=1)
    assert backend.to_numpy(axis_result).tolist() == [[20, 10], [40, 30]]

    flat_result = backend.take([[10, 20], [30, 40]], [3, 0], axis=None)
    assert backend.to_numpy(flat_result).tolist() == [40, 10]
