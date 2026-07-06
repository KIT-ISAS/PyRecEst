import numpy as np

import pyrecest.numerics as numerics
from pyrecest.numerics import assert_covariance_matrix


def test_numerics_all_includes_covariance_validator():
    assert "assert_covariance_matrix" in numerics.__all__
    assert numerics.assert_covariance_matrix is assert_covariance_matrix
    np.testing.assert_array_equal(assert_covariance_matrix(np.eye(2), dim=2), np.eye(2))
