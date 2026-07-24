import numpy as np
import pytest
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
    symmetrize_matrix,
)


def test_covariance_helpers_reject_cyclic_object_matrix():
    matrix = np.empty((2, 2), dtype=object)
    matrix[:] = 0.0
    matrix[0, 0] = matrix

    assert not is_symmetric(matrix)
    assert not is_positive_semidefinite(matrix)

    with pytest.raises(ValueError, match="covariance must contain numeric values"):
        assert_covariance_matrix(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        symmetrize_matrix(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        nearest_symmetric_psd(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        jittered_cholesky(matrix)


def test_covariance_helpers_accept_shared_acyclic_nested_scalars():
    one = np.array(1.0)
    zero = np.array(0.0)
    matrix = np.empty((2, 2), dtype=object)
    matrix[0, 0] = one
    matrix[0, 1] = zero
    matrix[1, 0] = zero
    matrix[1, 1] = one

    assert is_symmetric(matrix)
    assert is_positive_semidefinite(matrix)
    np.testing.assert_array_equal(
        np.asarray(assert_covariance_matrix(matrix)), np.eye(2)
    )
