import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_round_accepts_array_like_inputs_decimals_and_out():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend


def assert_close_list(actual, expected):
    actual = [float(value) for value in actual]
    assert len(actual) == len(expected)
    assert all(abs(one_actual - one_expected) < 1e-6 for one_actual, one_expected in zip(actual, expected))


for round_backend in (backend, raw_backend):
    result = round_backend.round([1.24, 2.76], decimals=1)
    assert_close_list(round_backend.to_numpy(result).tolist(), [1.2, 2.8])

    tensor = backend.array([1.24, 2.76])
    tensor_result = round_backend.round(tensor, decimals=1)
    assert_close_list(round_backend.to_numpy(tensor_result).tolist(), [1.2, 2.8])

    out = round_backend.array([0.0, 0.0])
    returned = round_backend.round([1.24, 2.76], decimals=1, out=out)
    assert returned is out
    assert_close_list(round_backend.to_numpy(out).tolist(), [1.2, 2.8])

try:
    backend.round([1.0], decimals=1.5)
except TypeError:
    pass
else:
    raise AssertionError("round accepted a non-integer decimals argument")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_pytorch_round_accepts_array_like_inputs_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "numpy"


def assert_close_list(actual, expected):
    actual = [float(value) for value in actual]
    assert len(actual) == len(expected)
    assert all(abs(one_actual - one_expected) < 1e-6 for one_actual, one_expected in zip(actual, expected))


result = raw_backend.round([1.24, 2.76], decimals=1)
assert_close_list(raw_backend.to_numpy(result).tolist(), [1.2, 2.8])

out = raw_backend.array([0.0, 0.0])
returned = raw_backend.round([1.24, 2.76], decimals=1, out=out)
assert returned is out
assert_close_list(raw_backend.to_numpy(out).tolist(), [1.2, 2.8])

try:
    raw_backend.round([1.0], decimals=1.5)
except TypeError:
    pass
else:
    raise AssertionError("round accepted a non-integer decimals argument")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
