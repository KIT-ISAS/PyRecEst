"""Regression tests for nested batches in hyperspherical uniform PDFs."""

import unittest

import numpy.testing as npt
from pyrecest.backend import array, ones
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    HypersphericalUniformDistribution,
)


class HypersphericalUniformNestedBatchPdfTest(unittest.TestCase):
    def test_pdf_preserves_all_leading_batch_dimensions(self):
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

        values = dist.pdf(points)

        expected_density = 1.0 / (
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(2)
        )
        self.assertEqual(values.shape, (2, 3))
        npt.assert_allclose(values, expected_density * ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
