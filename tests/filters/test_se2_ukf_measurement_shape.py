import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="SE2UKF update is not supported on JAX",
)
class TestSE2UKFMeasurementShape(unittest.TestCase):
    def test_update_identity_rejects_nonvector_measurements(self):
        measurement_noise = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]),
            eye(4) * 0.01,
        )
        invalid_measurements = (
            array([[1.0, 0.0, 0.0, 0.0]]),
            array([[1.0], [0.0], [0.0], [0.0]]),
            array([[1.0, 0.0], [0.0, 0.0]]),
        )

        for measurement in invalid_measurements:
            with self.subTest(shape=tuple(measurement.shape)):
                filter_instance = SE2UKF()
                with self.assertRaisesRegex(ValueError, "4-D vector"):
                    filter_instance.update_identity(measurement_noise, measurement)


if __name__ == "__main__":
    unittest.main()
