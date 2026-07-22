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
        lambda_ = array(2.0)
        kappa = array(1.3)
        distribution = WrappedLaplaceDistribution(lambda_, kappa)
        evaluation_points = array([0.0, 0.5, 1.0])
        expected_pdf = to_numpy(distribution.pdf(evaluation_points)).copy()

        lambda_[...] = 4.0
        kappa[...] = 0.75

        npt.assert_allclose(to_numpy(lambda_), 4.0)
        npt.assert_allclose(to_numpy(kappa), 0.75)
        npt.assert_allclose(to_numpy(distribution.lambda_), 2.0)
        npt.assert_allclose(to_numpy(distribution.kappa), 1.3)
        npt.assert_allclose(
            to_numpy(distribution.pdf(evaluation_points)),
            expected_pdf,
        )


if __name__ == "__main__":
    unittest.main()
