import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


def test_pdf_remains_finite_and_positive_for_tiny_gamma():
    gamma = 1.0e-20
    dist = WrappedCauchyDistribution(0.0, gamma)

    values = np.asarray(to_numpy(dist.pdf(array([0.0, 0.2, np.pi]))), dtype=float)

    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    npt.assert_allclose(
        values[0],
        1.0 / (2.0 * np.pi * np.tanh(gamma / 2.0)),
        rtol=1.0e-6,
    )


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Wrapped Cauchy CDF is not supported on this backend",
)
def test_cdf_remains_finite_at_mode_when_tiny_gamma_overflows_coth():
    gamma = 1.0e-308
    dist = WrappedCauchyDistribution(1.0, gamma)

    values = np.asarray(
        to_numpy(dist.cdf(array([1.0]), starting_point=0.0)),
        dtype=float,
    )

    assert np.all(np.isfinite(values))
    npt.assert_allclose(values, [0.5], atol=1.0e-12)
