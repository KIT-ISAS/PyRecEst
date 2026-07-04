"""Regression tests for PyTorch min/max device selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

from pyrecest.backend_support._pytorch_minmax_device_contract import (
    _preferred_pytorch_device,
)


@dataclass(frozen=True)
class _FakeDevice:
    type: str


@dataclass(frozen=True)
class _FakeTensor:
    device: _FakeDevice


class _FakeTorch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)


def test_preferred_pytorch_device_keeps_meta_operands_symbolic():
    cuda_value = _FakeTensor(_FakeDevice("cuda"))
    meta_value = _FakeTensor(_FakeDevice("meta"))

    assert _preferred_pytorch_device(_FakeTorch, cuda_value, meta_value).type == "meta"
