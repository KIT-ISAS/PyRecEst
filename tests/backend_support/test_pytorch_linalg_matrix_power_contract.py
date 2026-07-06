"""Regression tests for PyTorch linalg.matrix_power scalar exponent validation."""

import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_matrix_power_rejects_boolean_exponents():
    pytest.importorskip("torch")

    code = """
import numpy as np
import torch

from pyrecest.backend import linalg

matrix = [[1.0, 1.0], [0.0, 1.0]]
boolean_exponents = [
    True,
    False,
    np.bool_(True),
    np.array(True),
    torch.tensor(True),
]

for exponent in boolean_exponents:
    try:
        linalg.matrix_power(matrix, exponent)
    except TypeError:
        continue
    raise AssertionError(f"accepted boolean exponent {exponent!r}")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr


def test_pytorch_matrix_power_still_accepts_integer_scalars():
    pytest.importorskip("torch")

    code = """
import numpy as np

import pyrecest.backend as backend
from pyrecest.backend import linalg

matrix = [[1.0, 1.0], [0.0, 1.0]]
expected = [[1.0, 2.0], [0.0, 1.0]]

for exponent in [2, np.int64(2), np.array(2)]:
    result = linalg.matrix_power(matrix, exponent)
    assert backend.to_numpy(result).tolist() == expected
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
