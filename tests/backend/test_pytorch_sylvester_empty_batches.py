import pytest

pytest.importorskip("torch")

from pyrecest._backend import pytorch as pytorch_backend


@pytest.mark.parametrize(
    "dtype",
    (
        pytorch_backend.float32,
        pytorch_backend.complex64,
    ),
)
def test_sylvester_preserves_broadcast_empty_batches(dtype):
    a = pytorch_backend.empty((2, 0, 1, 2, 2), dtype=dtype)
    b = pytorch_backend.empty((1, 0, 4, 3, 3), dtype=dtype)
    q = pytorch_backend.empty((2, 1, 4, 2, 3), dtype=dtype)

    result = pytorch_backend.linalg.solve_sylvester(a, b, q)

    assert tuple(result.shape) == (2, 0, 4, 2, 3)
    assert result.dtype == dtype
    assert result.device == q.device


def test_sylvester_rejects_empty_batches_with_invalid_core_shapes():
    a = pytorch_backend.empty((0, 2, 3), dtype=pytorch_backend.float32)
    b = pytorch_backend.empty((0, 4, 4), dtype=pytorch_backend.float32)
    q = pytorch_backend.empty((0, 2, 4), dtype=pytorch_backend.float32)

    with pytest.raises(ValueError):
        pytorch_backend.linalg.solve_sylvester(a, b, q)
