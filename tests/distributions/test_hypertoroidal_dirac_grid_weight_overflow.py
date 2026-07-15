import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

from pyrecest.backend import array, to_numpy, zeros_like
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    HypertoroidalDiracDistribution,
)


class _ExtremeGridDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        super().__init__(2)
        maximum = np.finfo(np.float64).max
        self.grid_values = array(
            np.array([maximum, maximum / 2.0], dtype=np.float64)
        )
        self._grid = array([[0.1, 0.2], [0.3, 0.4]])

    def get_grid(self):
        return self._grid

    def pdf(self, xs):
        return zeros_like(xs)

    def sample(self, _n):
        raise AssertionError("grid conversion must not sample")


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="This regression requires float64 values near the representable limit.",
)
class HypertoroidalDiracGridWeightOverflowTest(unittest.TestCase):
    def test_grid_conversion_preserves_extreme_finite_weight_ratios(self):
        source = _ExtremeGridDistribution()
        source_weights = to_numpy(source.grid_values)

        self.assertTrue(np.all(np.isfinite(source_weights)))
        with np.errstate(over="ignore"):
            self.assertTrue(np.isinf(np.sum(source_weights)))

        converted = HypertoroidalDiracDistribution.from_distribution(source)

        npt.assert_allclose(
            to_numpy(converted.w),
            np.array([2.0 / 3.0, 1.0 / 3.0]),
            rtol=1.0e-12,
            atol=0.0,
        )
        npt.assert_allclose(to_numpy(converted.d), to_numpy(source.get_grid()))


if __name__ == "__main__":
    unittest.main()
