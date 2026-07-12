"""Regression tests for hyperspherical-uniform log-density input shapes."""

import unittest

import numpy.testing as npt
from pyrecest.backend import array, ones
from pyrecest.distributions import HypersphericalUniformDistribution


class HypersphericalUniformLnPdfBatchShapeTest(unittest.TestCase):
    def test_ln_pdf_preserves_all_leading_batch_dimensions(self):
        dist = HypersphericalUniformDistribution(2)
        points = array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
            ]
        )

        values = dist.ln_pdf(points)
        expected = -dist.get_ln_manifold_size() * ones((2, 3))

        self.assertEqual(values.shape, (2, 3))
        npt.assert_allclose(values, expected)

    def test_ln_pdf_rejects_scalar_input_with_value_error(self):
        dist = HypersphericalUniformDistribution(2)

        with self.assertRaisesRegex(ValueError, "shape"):
            dist.ln_pdf(1.0)


if __name__ == "__main__":
    unittest.main()
