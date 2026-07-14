import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, ones, pi
from pyrecest.distributions import CircularDiracDistribution, VonMisesDistribution
from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)


class TestCircularDiracDistribution(unittest.TestCase):
    def test_column_vector_locations_are_stored_flat(self):
        d = array([[0.0], [pi / 2.0], [pi]])
        w = ones(3) / 3.0

        wd = CircularDiracDistribution(d, w)

        self.assertEqual(wd.d.shape, (3,))
        npt.assert_allclose(wd.d, d[:, 0])
        npt.assert_allclose(wd.w, w)

    def test_column_vector_locations_with_uniform_weights_are_stored_flat(self):
        d = array([[0.0], [pi / 2.0], [pi]])

        wd = CircularDiracDistribution(d)

        self.assertEqual(wd.d.shape, (3,))
        self.assertEqual(wd.w.shape, (3,))
        npt.assert_allclose(wd.d, d[:, 0])

    def test_rejects_multidimensional_locations_for_circular_dirac(self):
        with self.assertRaisesRegex(ValueError, "shapes of d and w"):
            CircularDiracDistribution(array([[0.0, pi / 2.0], [pi, 3.0 * pi / 2.0]]))

    def test_from_distribution_preserves_circular_dirac_type(self):
        n_particles = 5
        vm = VonMisesDistribution(array(0.2), array(1.5))

        wd = CircularDiracDistribution.from_distribution(vm, n_particles)

        self.assertIsInstance(wd, CircularDiracDistribution)
        self.assertEqual(wd.d.shape, (n_particles,))
        self.assertEqual(wd.w.shape, (n_particles,))
        npt.assert_allclose(wd.w, ones(n_particles) / n_particles)

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "numpy",
        reason="Regression exercises NumPy float64 overflow semantics.",
    )
    def test_from_grid_distribution_scales_large_finite_weights(self):
        largest = np.finfo(float).max
        grid_distribution = CircularGridDistribution(array([largest, largest]))

        with np.errstate(over="raise", invalid="raise"):
            wd = CircularDiracDistribution.from_distribution(grid_distribution)

        self.assertIsInstance(wd, CircularDiracDistribution)
        npt.assert_allclose(wd.w, ones(2) / 2.0)


if __name__ == "__main__":
    unittest.main()
