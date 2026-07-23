import unittest

import numpy.testing as npt
from pyrecest.backend import asarray, ones, pi
from pyrecest.distributions.conditional.td_cond_td_grid_distribution import (
    TdCondTdGridDistribution,
)


class TdCondTdGridDistributionMultidimensionalTest(unittest.TestCase):
    def test_from_function_uses_total_cartesian_grid_size(self):
        """T2 x T2 factories must build one matrix row/column per grid point."""
        n_per_dimension = 2
        n_grid_points = n_per_dimension**2
        density = 1.0 / (2.0 * pi) ** 2

        def pairwise_conditional(a, _b):
            return ones(a.shape[0]) * density

        def cartesian_conditional(a, b):
            return ones((a.shape[0], b.shape[0])) * density

        cases = (
            (pairwise_conditional, False),
            (cartesian_conditional, True),
        )
        for conditional, does_cartesian_product in cases:
            with self.subTest(does_cartesian_product=does_cartesian_product):
                distribution = TdCondTdGridDistribution.from_function(
                    conditional,
                    n_per_dimension,
                    fun_does_cartesian_product=does_cartesian_product,
                    dim=4,
                )

                self.assertEqual(distribution.dim, 4)
                self.assertEqual(asarray(distribution.grid).shape, (n_grid_points, 2))
                self.assertEqual(
                    asarray(distribution.grid_values).shape,
                    (n_grid_points, n_grid_points),
                )
                npt.assert_allclose(distribution.grid_values, density)


if __name__ == "__main__":
    unittest.main()
