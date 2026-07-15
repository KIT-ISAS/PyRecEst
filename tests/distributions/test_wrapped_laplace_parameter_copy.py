import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


class WrappedLaplaceParameterCopyTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="JAX arrays are immutable and cannot expose caller-side aliasing.",
    )
    def test_constructor_owns_mutable_parameter_storage(self):
        rate = array(2.0)
        asymmetry = array(3.0)
        distribution = WrappedLaplaceDistribution(rate, asymmetry)
        evaluation_points = array([0.0, 0.5, 1.0])
        expected_pdf = to_numpy(distribution.pdf(evaluation_points)).copy()
        expected_moment = to_numpy(distribution.trigonometric_moment(1)).copy()

        rate[...] = 4.0
        asymmetry[...] = 0.5

        npt.assert_allclose(to_numpy(rate), 4.0)
        npt.assert_allclose(to_numpy(asymmetry), 0.5)
        npt.assert_allclose(to_numpy(distribution.lambda_), 2.0)
        npt.assert_allclose(to_numpy(distribution.kappa), 3.0)
        npt.assert_allclose(
            to_numpy(distribution.pdf(evaluation_points)),
            expected_pdf,
        )
        npt.assert_allclose(
            to_numpy(distribution.trigonometric_moment(1)),
            expected_moment,
        )


if __name__ == "__main__":
    unittest.main()
