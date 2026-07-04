"""Regression tests for PyTorch linspace device preservation."""

from __future__ import annotations

import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _linspace_device_contract_code(target_module: str) -> str:
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


def _non_cpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("meta")


target = {target_module}
device = _non_cpu_device()

scalar_result = target.linspace(torch.tensor(0.0), torch.tensor(1.0, device=device), num=3)
assert scalar_result.device.type == device.type
assert tuple(scalar_result.shape) == (3,)
if device.type != "meta":
    assert torch.allclose(scalar_result.cpu(), torch.tensor([0.0, 0.5, 1.0]))

batched_result = target.linspace(
    torch.tensor([0, 10]),
    torch.tensor([2, 14], device=device),
    num=3,
)
assert batched_result.device.type == device.type
assert tuple(batched_result.shape) == (3, 2)
if device.type != "meta":
    expected = torch.tensor([[0.0, 10.0], [1.0, 12.0], [2.0, 14.0]])
    assert torch.allclose(batched_result.cpu(), expected)

print("ok")
"""


def test_raw_pytorch_linspace_prefers_existing_non_cpu_endpoint_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _linspace_device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_linspace_prefers_existing_non_cpu_endpoint_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _linspace_device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
