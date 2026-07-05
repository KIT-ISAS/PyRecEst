"""Regression tests for PyTorch stack/concatenate device normalization."""

from __future__ import annotations

import pytest

from tests.support.backend_runner import run_backend_code


_STACK_CONCAT_META_DEVICE_CHECK = """
import pyrecest
import torch

meta = torch.empty((1,), device="meta")

{backend_import}

stacked = backend.stack(([1.0], meta), axis=0)
concatenated = backend.concatenate(([1.0], meta), axis=0)

assert stacked.device.type == "meta"
assert tuple(stacked.shape) == (2, 1)
assert concatenated.device.type == "meta"
assert tuple(concatenated.shape) == (2,)
"""


def test_raw_pytorch_stack_and_concatenate_preserve_meta_device():
    pytest.importorskip("torch")

    code = _STACK_CONCAT_META_DEVICE_CHECK.format(
        backend_import="import pyrecest._backend.pytorch as backend"
    )
    result = run_backend_code("numpy", code)

    assert result.returncode == 0, result.stderr


def test_public_pytorch_stack_and_concatenate_preserve_meta_device():
    pytest.importorskip("torch")

    code = _STACK_CONCAT_META_DEVICE_CHECK.format(
        backend_import="import pyrecest.backend as backend"
    )
    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
