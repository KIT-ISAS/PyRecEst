import numpy as np
import pytest

from pyrecest.exceptions import NumericalStabilityError
from pyrecest.numerics import jittered_cholesky


def test_jittered_cholesky_rejects_factor_after_jitter_overflow():
    max_float = np.finfo(float).max
    matrix = np.diag([max_float / 2.0, -1.0])

    with pytest.raises(
        NumericalStabilityError, match="Cholesky factorization failed"
    ):
        jittered_cholesky(
            matrix,
            initial_jitter=max_float,
            max_attempts=1,
        )
