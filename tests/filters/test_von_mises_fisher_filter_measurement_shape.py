import unittest

from pyrecest.backend import array
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.filters.von_mises_fisher_filter import VonMisesFisherFilter


class TestVonMisesFisherFilterMeasurementShape(unittest.TestCase):
    def test_update_identity_rejects_nonvector_measurements(self):
        vmf_filter = VonMisesFisherFilter()
        vmf_filter.filter_state = VonMisesFisherDistribution(array([1.0, 0.0]), 0.7)
        measurement_noise = VonMisesFisherDistribution(array([0.0, 1.0]), 0.9)

        for measurement in ([[1.0, 0.0]], [[1.0], [0.0]]):
            with self.subTest(measurement=measurement):
                with self.assertRaisesRegex(ValueError, "shape"):
                    vmf_filter.update_identity(measurement_noise, measurement)


if __name__ == "__main__":
    unittest.main()
