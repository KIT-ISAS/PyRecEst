import importlib.util
import os
import subprocess
import sys

import pytest


def _run_python(code, *, backend_name=None):
    env = os.environ.copy()
    if backend_name is None:
        env.pop("PYRECEST_BACKEND", None)
    else:
        env["PYRECEST_BACKEND"] = backend_name

    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_jax_triangular_vector_helpers_are_patched_under_default_backend():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    _run_python(
        r"""
import pyrecest  # noqa: F401
import pyrecest.backend as public_backend
import pyrecest._backend.jax as raw_backend

assert public_backend.__backend_name__ == "numpy"

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
batched_matrix = [
    matrix,
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
]


def as_list(value):
    converted = raw_backend.to_numpy(value)
    return converted.tolist() if hasattr(converted, "tolist") else converted


assert as_list(raw_backend.tril_to_vec(matrix)) == [1, 4, 5, 7, 8, 9]
assert as_list(raw_backend.triu_to_vec(matrix)) == [1, 2, 3, 5, 6, 9]
assert as_list(raw_backend.tril_to_vec(matrix, k=-1)) == [4, 7, 8]
assert as_list(raw_backend.triu_to_vec(matrix, k=1)) == [2, 3, 6]
assert as_list(raw_backend.tril_to_vec(batched_matrix)) == [
    [1, 4, 5, 7, 8, 9],
    [10, 13, 14, 16, 17, 18],
]
"""
    )


@pytest.mark.backend_portable
def test_public_jax_triangular_vector_helpers_accept_arraylike_inputs():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    _run_python(
        r"""
import pyrecest.backend as backend
import pyrecest._backend.jax as raw_backend

assert backend.__backend_name__ == "jax"

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
batched_matrix = [
    matrix,
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
]


def as_list(module, value):
    converted = module.to_numpy(value)
    return converted.tolist() if hasattr(converted, "tolist") else converted


for module in (backend, raw_backend):
    assert as_list(module, module.tril_to_vec(matrix)) == [1, 4, 5, 7, 8, 9]
    assert as_list(module, module.triu_to_vec(matrix)) == [1, 2, 3, 5, 6, 9]
    assert as_list(module, module.tril_to_vec(matrix, k=-1)) == [4, 7, 8]
    assert as_list(module, module.triu_to_vec(matrix, k=1)) == [2, 3, 6]
    assert as_list(module, module.tril_to_vec(batched_matrix)) == [
        [1, 4, 5, 7, 8, 9],
        [10, 13, 14, 16, 17, 18],
    ]
""",
        backend_name="jax",
    )
