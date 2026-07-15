import numpy as np

from pyrecest.numerics import (
    jittered_cholesky,
    nearest_symmetric_psd,
    symmetrize_matrix,
)


def test_covariance_repairs_preserve_maximum_finite_diagonal():
    maximum = np.finfo(float).max
    matrix = np.diag([maximum, maximum])

    with np.errstate(over="ignore", invalid="ignore"):
        symmetric = np.asarray(symmetrize_matrix(matrix))
        repaired = np.asarray(nearest_symmetric_psd(matrix))
        factor, jitter = jittered_cholesky(matrix)
        factor = np.asarray(factor)

    assert np.all(np.isfinite(symmetric))
    np.testing.assert_array_equal(symmetric, matrix)

    assert np.all(np.isfinite(repaired))
    np.testing.assert_allclose(repaired, matrix, rtol=2e-15, atol=0.0)

    assert jitter == 0.0
    assert np.all(np.isfinite(factor))
    np.testing.assert_allclose(factor @ factor.T, matrix, rtol=2e-15, atol=0.0)
