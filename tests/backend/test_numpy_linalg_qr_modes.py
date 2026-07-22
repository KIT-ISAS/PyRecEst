import numpy as np
import pytest
from pyrecest._backend.numpy import linalg


def test_qr_mode_r_returns_single_batched_r_factor():
    matrices = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]),
            np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 4.0]]),
        ]
    )

    result = linalg.qr(matrices, mode="r")
    expected = np.linalg.qr(matrices, mode="r")

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)


def test_qr_mode_raw_preserves_batched_numpy_contract():
    matrices = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]),
            np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 4.0]]),
        ]
    )

    h, tau = linalg.qr(matrices, mode="raw")
    expected_h, expected_tau = np.linalg.qr(matrices, mode="raw")

    np.testing.assert_allclose(h, expected_h)
    np.testing.assert_allclose(tau, expected_tau)


@pytest.mark.parametrize("mode", ["reduced", "complete", "r", "raw"])
def test_qr_empty_batches_preserve_numpy_contract(mode):
    matrices = np.empty((0, 3, 2))

    result = linalg.qr(matrices, mode=mode)
    expected = np.linalg.qr(matrices, mode=mode)

    if isinstance(expected, tuple):
        assert isinstance(result, tuple)
        assert len(result) == len(expected)
        for actual_factor, expected_factor in zip(result, expected):
            assert actual_factor.shape == expected_factor.shape
            np.testing.assert_allclose(actual_factor, expected_factor)
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected)
