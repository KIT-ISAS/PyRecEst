import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_public_jax_argsort_accepts_array_like_inputs():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend

assert backend.__backend_name__ == "jax"

axis_result = backend.argsort([[3, 1, 2], [0, 5, 4]], axis=1)
assert backend.to_numpy(axis_result).tolist() == [[1, 2, 0], [0, 2, 1]]

flat_result = backend.argsort([[3, 1], [0, 2]], axis=None)
assert backend.to_numpy(flat_result).tolist() == [2, 1, 3, 0]

dim_result = backend.argsort([[3, 1], [0, 2]], dim=0)
assert backend.to_numpy(dim_result).tolist() == [[1, 0], [0, 1]]

descending_result = backend.argsort([[3, 1], [0, 2]], axis=None, descending=True)
assert backend.to_numpy(descending_result).tolist() == [0, 3, 1, 2]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_jax_argsort_accepts_array_like_inputs_after_import():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest  # noqa: F401  # triggers backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.jax as raw_jax

assert backend.__backend_name__ == "numpy"

axis_result = raw_jax.argsort([[3, 1, 2], [0, 5, 4]], axis=1)
assert raw_jax.to_numpy(axis_result).tolist() == [[1, 2, 0], [0, 2, 1]]

flat_result = raw_jax.argsort([[3, 1], [0, 2]], axis=None)
assert raw_jax.to_numpy(flat_result).tolist() == [2, 1, 3, 0]

dim_result = raw_jax.argsort([[3, 1], [0, 2]], dim=0)
assert raw_jax.to_numpy(dim_result).tolist() == [[1, 0], [0, 1]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
