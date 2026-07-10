"""Regression tests for empty GlobalNearestNeighbor measurement updates."""

import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class TestGlobalNearestNeighborEmptyMeasurements(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Nearest-neighbor trackers are not supported on this backend",
    )
    def test_empty_per_measurement_covariance_tensor_is_a_missed_update(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = [KalmanFilter(GaussianDistribution(zeros(2), eye(2)))]
        prior_mean = tracker.filter_bank[0].filter_state.mu.copy()
        prior_covariance = tracker.filter_bank[0].filter_state.C.copy()

        measurements = zeros((2, 0))
        covariances = zeros((2, 2, 0))

        with self.assertWarnsRegex(UserWarning, "No measurement"):
            association = tracker.find_association(
                measurements,
                eye(2),
                covariances,
            )
        npt.assert_array_equal(association, [0])

        with self.assertWarnsRegex(UserWarning, "No measurement"):
            tracker.update_linear(measurements, eye(2), covariances)

        npt.assert_array_equal(tracker.filter_bank[0].filter_state.mu, prior_mean)
        npt.assert_array_equal(
            tracker.filter_bank[0].filter_state.C,
            prior_covariance,
        )


if __name__ == "__main__":
    unittest.main()
