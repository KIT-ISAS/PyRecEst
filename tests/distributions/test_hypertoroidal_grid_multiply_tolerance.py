import unittest

import numpy.testing as npt

from pyrecest.backend import array, mean, pi
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)


class HypertoroidalGridMultiplyToleranceTest(unittest.TestCase):
    def test_multiply_accepts_numerically_equivalent_custom_grids(self):
        grid = array([[0.0], [pi]])
        perturbed_grid = array([[0.0], [pi + 1.0e-12]])
        left = HypertoroidalGridDistribution(array([1.0, 2.0]), grid=grid)
        right = HypertoroidalGridDistribution(
            array([3.0, 4.0]), grid=perturbed_grid
        )

        product = left.multiply(right)

        unnormalized = left.grid_values * right.grid_values
        expected = unnormalized / (2.0 * pi * mean(unnormalized))
        npt.assert_allclose(product.grid_values, expected)
        npt.assert_allclose(product.get_grid(), grid)


if __name__ == "__main__":
    unittest.main()
