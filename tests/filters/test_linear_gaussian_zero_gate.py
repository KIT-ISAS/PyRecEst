import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, array, eye, to_numpy
from pyrecest.filters._linear_gaussian import linear_gaussian_update_robust


@unittest.skipIf(
    __backend_name__ in ("pytorch", "jax"),
    reason="tests compare backend arrays with NumPy helpers",
)
class LinearGaussianZeroGateThresholdTest(unittest.TestCase):
    def setUp(self):
        self.mean = array([0.0, 0.0])
        self.covariance = eye(2)
        self.measurement_matrix = eye(2)
        self.meas_noise = eye(2)

    def test_zero_hard_gate_threshold_rejects_nonzero_nis(self):
        updated_mean, updated_covariance, diagnostics = linear_gaussian_update_robust(
            self.mean,
            self.covariance,
            array([1.0, 0.0]),
            self.measurement_matrix,
            self.meas_noise,
            robust_update="none",
            gate_threshold=0.0,
            return_diagnostics=True,
        )

        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(diagnostics["action"], "rejected")
        self.assertEqual(float(diagnostics["scale"]), 1.0)
        npt.assert_allclose(to_numpy(updated_mean), to_numpy(self.mean))
        npt.assert_allclose(to_numpy(updated_covariance), to_numpy(self.covariance))

    def test_zero_hard_gate_threshold_accepts_zero_nis(self):
        updated_mean, updated_covariance, diagnostics = linear_gaussian_update_robust(
            self.mean,
            self.covariance,
            array([0.0, 0.0]),
            self.measurement_matrix,
            self.meas_noise,
            robust_update="none",
            gate_threshold=0.0,
            return_diagnostics=True,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(diagnostics["action"], "updated")
        self.assertEqual(float(diagnostics["nis"]), 0.0)
        npt.assert_allclose(to_numpy(updated_mean), to_numpy(self.mean))
        npt.assert_allclose(to_numpy(updated_covariance), to_numpy(self.covariance))


if __name__ == "__main__":
    unittest.main()
