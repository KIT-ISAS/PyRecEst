import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_dot_supports_boolean_operands():
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
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

cases = [
    ([True, False, True], [False, True, True], True),
    (
        [[True, False, False], [False, False, False]],
        [[True, True, False], [True, True, True]],
        [True, False],
    ),
    (True, [False, True], [False, True]),
]

for left, right, expected in cases:
    public_result = backend.dot(backend.array(left), backend.array(right))
    raw_result = raw_pytorch.dot(raw_pytorch.array(left), raw_pytorch.array(right))

    assert public_result.dtype == torch.bool
    assert raw_result.dtype == torch.bool
    assert public_result.tolist() == expected
    assert raw_result.tolist() == expected
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
