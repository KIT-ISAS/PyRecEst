import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, asarray, to_numpy
from pyrecest.sampling import MerweScaledSigmaPoints


@unittest.skipIf(
    __backend_name__ == "pytorch",
    reason="Sigma-point tests use NumPy assertions and the PyTorch backend is unsupported",
)
class TestMerweSigmaPointScaling(unittest.TestCase):
    def test_small_alpha_uses_direct_positive_scale(self):
        alpha = 1.0e-10
        points = MerweScaledSigmaPoints(n=2, alpha=alpha, beta=2.0, kappa=0.0)

        sigmas = to_numpy(points.sigma_points(asarray(np.zeros(2)), asarray(np.eye(2))))
        step = np.sqrt(alpha * alpha * 2.0)
        expected = np.array(
            [
                [0.0, 0.0],
                [step, 0.0],
                [0.0, step],
                [-step, 0.0],
                [0.0, -step],
            ]
        )

        self.assertTrue(np.all(np.isfinite(to_numpy(points.Wm))))
        self.assertTrue(np.all(np.isfinite(to_numpy(points.Wc))))
        npt.assert_allclose(sigmas, expected, rtol=1.0e-12, atol=0.0)

    def test_rejects_underflowing_and_overflowing_scale(self):
        for alpha in (1.0e-200, 1.0e200):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(
                ValueError, "alpha scaling factor must be finite and positive"
            ):
                MerweScaledSigmaPoints(n=2, alpha=alpha, beta=2.0, kappa=0.0)


if __name__ == "__main__":
    unittest.main()
