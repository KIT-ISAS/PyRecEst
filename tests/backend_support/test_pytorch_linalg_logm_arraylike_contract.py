import pytest

import pyrecest.backend as backend
from pyrecest.backend import linalg as backend_linalg

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")
pytorch_linalg = pytest.importorskip("pyrecest._backend.pytorch.linalg")


def _as_list(value):
    return pytorch_backend.to_numpy(value).tolist()


def test_raw_pytorch_logm_accepts_array_like_integer_inputs():
    result = pytorch_linalg.logm([[1, 0], [0, 1]])

    assert result.dtype.is_floating_point
    assert _as_list(result) == [[0.0, 0.0], [0.0, 0.0]]


def test_public_pytorch_logm_accepts_array_like_integer_inputs_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    result = backend_linalg.logm([[1, 0], [0, 1]])

    assert result.dtype.is_floating_point
    assert _as_list(result) == [[0.0, 0.0], [0.0, 0.0]]
