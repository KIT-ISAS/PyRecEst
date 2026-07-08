import importlib.util
import os
import subprocess
import sys

import pytest


def _subprocess_env(selected_backend):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = selected_backend
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_public_pytorch_like_creation_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend

source = [[1, 2], [3, 4]]

zeros = backend.zeros_like(source)
assert backend.to_numpy(zeros).tolist() == [[0, 0], [0, 0]]

ones = backend.ones_like(source)
assert backend.to_numpy(ones).tolist() == [[1, 1], [1, 1]]

filled = backend.full_like(source, 7)
assert backend.to_numpy(filled).tolist() == [[7, 7], [7, 7]]

empty = backend.empty_like(source)
assert tuple(empty.shape) == (2, 2)
assert empty.dtype == zeros.dtype

typed = backend.zeros_like(source, dtype="torch.float64")
assert str(typed.dtype) == "torch.float64"

print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_subprocess_env("pytorch"),
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_like_creation_is_patched_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert backend.__backend_name__ == "numpy"
source = [[1, 2], [3, 4]]

zeros = raw_pytorch.zeros_like(source)
assert raw_pytorch.to_numpy(zeros).tolist() == [[0, 0], [0, 0]]

ones = raw_pytorch.ones_like(source)
assert raw_pytorch.to_numpy(ones).tolist() == [[1, 1], [1, 1]]

filled = raw_pytorch.full_like(source, 5)
assert raw_pytorch.to_numpy(filled).tolist() == [[5, 5], [5, 5]]

empty = raw_pytorch.empty_like(source)
assert tuple(empty.shape) == (2, 2)
assert empty.dtype == zeros.dtype

typed = raw_pytorch.zeros_like(source, dtype="torch.float64")
assert str(typed.dtype) == "torch.float64"

print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_subprocess_env("numpy"),
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
