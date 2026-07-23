import unittest

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.filters.bingham_filter import BinghamFilter


class TestBinghamFilterMeasurementShape(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_update_identity_rejects_nonvector_measurements(self):
        filter_2d = BinghamFilter()
        filter_2d.filter_state = BinghamDistribution(array([-5.0, 0.0]), eye(2))
        noise_2d = BinghamDistribution(array([-3.0, 0.0]), eye(2))

        filter_4d = BinghamFilter()
        noise_4d = BinghamDistribution(
            array([-2.0, -2.0, -2.0, 0.0]),
            eye(4),
        )

        invalid_measurements = (
            (filter_2d, noise_2d, [[1.0, 0.0]]),
            (filter_2d, noise_2d, [[1.0], [0.0]]),
            (filter_4d, noise_4d, [[1.0, 0.0], [0.0, 0.0]]),
        )
        for filter_instance, noise, measurement in invalid_measurements:
            with self.subTest(measurement=measurement):
                with self.assertRaisesRegex(ValueError, "shape"):
                    filter_instance.update_identity(noise, measurement)


if __name__ == "__main__":
    unittest.main()
