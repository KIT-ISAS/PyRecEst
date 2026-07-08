import numpy as np
import pytest
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
)


@pytest.mark.parametrize(
    "value",
    [
        np.timedelta64(1, "ns"),
        np.array(np.timedelta64(1, "ns"), dtype=object),
        np.datetime64("2020-01-01"),
        np.array(np.datetime64("2020-01-01"), dtype=object),
    ],
)
def test_numerical_scalar_controls_reject_temporal_values(value):
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="atol"):
        is_symmetric(matrix, atol=value)
    with pytest.raises(ValueError, match="atol"):
        is_positive_semidefinite(matrix, atol=value)
    with pytest.raises(ValueError, match="atol"):
        assert_covariance_matrix(matrix, atol=value)
    with pytest.raises(ValueError, match="min_eigenvalue"):
        nearest_symmetric_psd(matrix, min_eigenvalue=value)
    with pytest.raises(ValueError, match="initial_jitter"):
        jittered_cholesky(matrix, initial_jitter=value)
