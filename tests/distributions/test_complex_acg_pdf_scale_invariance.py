import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, complex128, isfinite, real
from pyrecest.distributions import ComplexAngularCentralGaussianDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Numerical underflow regression is evaluated with NumPy float64 semantics",
)
class TestComplexACGPDFScaleInvariance(unittest.TestCase):
    def test_pdf_is_scale_invariant_when_determinant_would_underflow(self):
        parameter = array(
            [[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            dtype=complex128,
        )
        scaled_parameter = 1e-200 * parameter
        inv_sqrt_two = 1.0 / np.sqrt(2.0)
        point = array([inv_sqrt_two + 0.0j, 1j * inv_sqrt_two])

        reference = ComplexAngularCentralGaussianDistribution(parameter).pdf(point)
        scaled = ComplexAngularCentralGaussianDistribution(scaled_parameter).pdf(point)

        self.assertTrue(bool(isfinite(scaled)))
        npt.assert_allclose(float(real(scaled)), float(real(reference)), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
