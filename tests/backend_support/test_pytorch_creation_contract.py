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
def test_pytorch_creation_helpers_accept_numpy_dtypes_and_shape_arrays():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

for creation_backend in (backend, raw_backend):
    zeros = creation_backend.zeros(np.array([2, 3]), dtype=np.float64)
    assert tuple(zeros.shape) == (2, 3)
    assert zeros.dtype == creation_backend.float64
    assert creation_backend.to_numpy(zeros).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    ones = creation_backend.ones(torch.tensor([2]), dtype=np.dtype("int64"))
    assert tuple(ones.shape) == (2,)
    assert ones.dtype == creation_backend.int64
    assert creation_backend.to_numpy(ones).tolist() == [1, 1]

    empty = creation_backend.empty(np.array(2), dtype=np.float32)
    assert tuple(empty.shape) == (2,)
    assert empty.dtype == creation_backend.float32

    full = creation_backend.full(np.array([2]), np.array(7), dtype=np.int64)
    assert tuple(full.shape) == (2,)
    assert full.dtype == creation_backend.int64
    assert creation_backend.to_numpy(full).tolist() == [7, 7]

    full_from_torch_scalar = creation_backend.full(
        np.array([2]), torch.tensor(8), dtype=np.int64
    )
    assert tuple(full_from_torch_scalar.shape) == (2,)
    assert full_from_torch_scalar.dtype == creation_backend.int64
    assert creation_backend.to_numpy(full_from_torch_scalar).tolist() == [8, 8]

    try:
        creation_backend.zeros(np.array([-1]))
    except ValueError:
        pass
    else:
        raise AssertionError("zeros accepted a negative dimension")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )


@pytest.mark.backend_portable
def test_raw_pytorch_creation_helpers_work_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

zeros = raw_backend.zeros(np.array([2]), dtype=np.float64)
assert tuple(zeros.shape) == (2,)
assert zeros.dtype == raw_backend.float64
assert raw_backend.to_numpy(zeros).tolist() == [0.0, 0.0]

ones = raw_backend.ones(torch.tensor([2, 1]), dtype=np.dtype("int64"))
assert tuple(ones.shape) == (2, 1)
assert ones.dtype == raw_backend.int64
assert raw_backend.to_numpy(ones).tolist() == [[1], [1]]

empty = raw_backend.empty(np.array(2), dtype=np.float32)
assert tuple(empty.shape) == (2,)
assert empty.dtype == raw_backend.float32

full = raw_backend.full(torch.tensor([2]), np.array(5), dtype=np.int64)
assert tuple(full.shape) == (2,)
assert full.dtype == raw_backend.int64
assert raw_backend.to_numpy(full).tolist() == [5, 5]

torch_scalar_full = raw_backend.full(np.array([2]), torch.tensor(6), dtype=np.int64)
assert tuple(torch_scalar_full.shape) == (2,)
assert torch_scalar_full.dtype == raw_backend.int64
assert raw_backend.to_numpy(torch_scalar_full).tolist() == [6, 6]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
