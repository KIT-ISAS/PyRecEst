import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


class LinearBoxParticleDistributionExtremeBoundsTest(unittest.TestCase):
    @staticmethod
    def _backend_dtype_and_maximum():
        backend_dtype = to_numpy(array([1.0])).dtype
        return backend_dtype, np.finfo(backend_dtype).max

    def test_centers_remain_finite_for_large_same_sign_bounds(self):
        backend_dtype, maximum = self._backend_dtype_and_maximum()
        dist = LinearBoxParticleDistribution(
            array([[maximum / 2.0]]),
            array([[maximum]]),
        )
        expected = np.asarray([[maximum / 2.0 + maximum / 4.0]], dtype=backend_dtype)

        centers = to_numpy(dist.centers())

        self.assertTrue(np.all(np.isfinite(centers)))
        npt.assert_allclose(centers, expected)
        npt.assert_allclose(to_numpy(dist.d), expected)
        npt.assert_allclose(to_numpy(dist.mean()), expected[0])
        npt.assert_allclose(to_numpy(dist.mode()), expected[0])

    def test_half_widths_remain_finite_for_opposite_extreme_bounds(self):
        backend_dtype, maximum = self._backend_dtype_and_maximum()
        dist = LinearBoxParticleDistribution(
            array([[-maximum]]),
            array([[maximum]]),
        )
        expected = np.asarray([[maximum]], dtype=backend_dtype)

        half_widths = to_numpy(dist.half_widths())

        self.assertTrue(np.all(np.isfinite(half_widths)))
        npt.assert_allclose(half_widths, expected)


if __name__ == "__main__":
    unittest.main()
