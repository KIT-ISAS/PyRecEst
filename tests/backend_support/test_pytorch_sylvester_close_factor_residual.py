"""Regression coverage for close-factor PyTorch Sylvester shortcuts."""

import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_sylvester_falls_back_when_close_factor_shortcut_is_inaccurate():
    pytest.importorskip("torch")
    code = """
import torch
from pyrecest.backend import linalg

# These positive-definite factors pass the shortcut's approximate equality
# check, but replacing B by A changes the ill-conditioned solution materially.
a = torch.diag(torch.tensor([1.0e-6, 1.0], dtype=torch.float64))
b = torch.diag(torch.tensor([1.5e-6, 1.0], dtype=torch.float64))
q = torch.eye(2, dtype=torch.float64)

assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)

solution = linalg.solve_sylvester(a, b, q)
residual = a @ solution + solution @ b - q

assert torch.linalg.norm(residual).item() < 1e-12
torch.testing.assert_close(
    solution[0, 0],
    torch.tensor(4.0e5, dtype=torch.float64),
    rtol=1e-12,
    atol=0.0,
)
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
