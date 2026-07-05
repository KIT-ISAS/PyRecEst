import importlib.util
import os
import subprocess
import sys

import pytest


def _pythonpath_env(backend_name):
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
def test_public_pytorch_split_rejects_noninteger_cut_indices():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend

parts = backend.split([0, 1, 2, 3, 4], [1, 3])
assert [backend.to_numpy(part).tolist() for part in parts] == [[0], [1, 2], [3, 4]]

for cut_indices in ([1.5], backend.array([1.5])):
    try:
        backend.split([0, 1, 2, 3], cut_indices)
    except TypeError:
        pass
    else:
        raise AssertionError("split accepted non-integer cut indices")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_pythonpath_env("pytorch"),
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_split_default_backend_rejects_noninteger_cut_indices():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert backend.__backend_name__ == "numpy"

parts = raw_pytorch.split([0, 1, 2, 3, 4], [2, 4])
assert [raw_pytorch.to_numpy(part).tolist() for part in parts] == [[0, 1], [2, 3], [4]]

for cut_indices in ([1.5], raw_pytorch.array([1.5])):
    try:
        raw_pytorch.split([0, 1, 2, 3], cut_indices)
    except TypeError:
        pass
    else:
        raise AssertionError("raw split accepted non-integer cut indices")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_pythonpath_env("numpy"),
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
