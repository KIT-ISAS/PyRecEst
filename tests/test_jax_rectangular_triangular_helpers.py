import pytest

pytest.importorskip("jax")


def _to_list(jax_backend, value):
    return jax_backend.to_numpy(value).tolist()


def test_raw_jax_triangular_helpers_use_rectangular_matrix_shape_after_import():
    import pyrecest  # noqa: F401
    import pyrecest._backend.jax as raw_jax

    values = [[1, 2, 3], [4, 5, 6]]

    assert _to_list(raw_jax, raw_jax.tril_to_vec(values)) == [1, 4, 5]
    assert _to_list(raw_jax, raw_jax.triu_to_vec(values)) == [1, 2, 3, 5, 6]


def test_public_jax_triangular_helpers_use_rectangular_matrix_shape_when_active():
    import pyrecest.backend as backend

    if getattr(backend, "__backend_name__", None) != "jax":
        pytest.skip("public JAX backend is not active")

    values = backend.asarray([[1, 2, 3], [4, 5, 6]])

    assert backend.to_numpy(backend.tril_to_vec(values)).tolist() == [1, 4, 5]
    assert backend.to_numpy(backend.triu_to_vec(values)).tolist() == [1, 2, 3, 5, 6]
