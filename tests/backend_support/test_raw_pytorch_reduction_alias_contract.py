import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_test_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_raw_pytorch_reductions_accept_dim_keepdim_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

values = raw_backend.array(
    [[[0, 1], [2, 0]], [[3, 4], [0, 0]]],
    dtype=raw_backend.int64,
)

assert raw_backend.max(values, dim=(0, 2), keepdim=True).tolist() == [[[4], [2]]]
assert raw_backend.min(values, dim=(0, 2), keepdim=True).tolist() == [[[0], [0]]]
assert raw_backend.prod(values + 1, dim=(0, 2), keepdim=True).tolist() == [[[40], [3]]]
assert raw_backend.count_nonzero(values, dim=(0, 2), keepdim=True).tolist() == [[[3], [1]]]
assert raw_backend.any(values, dim=(0, 2), keepdim=True).tolist() == [[[True], [True]]]
assert raw_backend.all(values >= 0, dim=(0, 2), keepdim=True).tolist() == [[[True], [True]]]
assert raw_backend.amax(values, dim=(0, 2), keepdim=True).tolist() == [[[4], [2]]]
assert raw_backend.amin(values, dim=(0, 2), keepdim=True).tolist() == [[[0], [0]]]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
