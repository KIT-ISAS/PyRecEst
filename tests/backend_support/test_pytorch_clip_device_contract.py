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
def test_pytorch_clip_prefers_non_cpu_bound_device_with_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend
import torch

assert getattr(backend, "__backend_name__", None) == "pytorch"
upper = torch.empty((2,), device="meta")

for clip_func in (backend.clip, pytorch_backend.clip):
    result = clip_func([0.0, 2.0], a_max=upper)
    assert result.device.type == "meta"
    assert tuple(result.shape) == (2,)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch"))


@pytest.mark.backend_portable
def test_raw_pytorch_clip_prefers_non_cpu_bound_device_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend
import torch

assert getattr(backend, "__backend_name__", None) == "numpy"
upper = torch.empty((2,), device="meta")

result = pytorch_backend.clip([0.0, 2.0], a_max=upper)
assert result.device.type == "meta"
assert tuple(result.shape) == (2,)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_backend_test_env("numpy"))
