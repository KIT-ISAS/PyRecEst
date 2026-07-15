import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.filters.relaxed_s3f_circular import grid_probability_masses


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="This regression requires float64 values near the representable limit.",
)
class RelaxedS3FCircularWeightOverflowTest(unittest.TestCase):
    def test_normalizes_finite_weights_with_overflowing_raw_sum(self):
        relative_weights = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        huge_weights = relative_weights * np.finfo(np.float64).max

        self.assertTrue(np.all(np.isfinite(huge_weights)))
        with np.errstate(over="ignore"):
            self.assertTrue(np.isinf(np.sum(huge_weights)))

        actual = to_numpy(grid_probability_masses(array(huge_weights)))
        expected = relative_weights / relative_weights.sum()

        npt.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)
        npt.assert_allclose(np.sum(actual), 1.0, rtol=1.0e-12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
