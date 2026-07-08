import unittest

import numpy as np
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter, NISGate
from pyrecest.filters.survival_aware_association import (
    SurvivalAwareAssociationConfig,
    build_survival_aware_linear_gaussian_hypothesis_associator,
    survival_aware_missed_detection_costs,
    survival_aware_track_log_prior,
)
from pyrecest.filters.track_manager import Track, TrackStatus


def _track(mean, *, track_id=0, hits=3, misses=0, existence=1.0):
    state = GaussianDistribution(np.array([float(mean)]), np.array([[0.2]]))
    return Track(
        track_id=track_id,
        single_target_filter=KalmanFilter(state),
        status=TrackStatus.CONFIRMED,
        hits=hits,
        misses=misses,
        age=max(1, hits + misses),
        metadata={"existence_probability": existence},
    )


class SurvivalAwareAssociationTest(unittest.TestCase):
    def test_prior_discounts_stale_tracks(self):
        config = SurvivalAwareAssociationConfig(
            survival_probability=0.8,
            detection_probability=0.9,
            visibility_probability=1.0,
            mass_decay=0.5,
        )
        fresh = _track(0.0, hits=4, misses=0)
        stale = _track(0.0, hits=4, misses=3)

        self.assertLess(
            survival_aware_track_log_prior(stale, config=config),
            survival_aware_track_log_prior(fresh, config=config),
        )

    def test_missed_detection_is_cheaper_when_visibility_is_low(self):
        track = _track(0.0, existence=0.9)
        high_visibility = SurvivalAwareAssociationConfig(
            detection_probability=0.9,
            visibility_probability=1.0,
        )
        low_visibility = SurvivalAwareAssociationConfig(
            detection_probability=0.9,
            visibility_probability=0.1,
        )

        high_visibility_cost = survival_aware_missed_detection_costs(
            [track],
            config=high_visibility,
        )[0]
        low_visibility_cost = survival_aware_missed_detection_costs(
            [track],
            config=low_visibility,
        )[0]

        self.assertLess(low_visibility_cost, high_visibility_cost)

    def test_track_manager_associator_keeps_far_measurement_unassigned(self):
        tracks = [_track(0.0, hits=5, misses=0, existence=0.95)]
        measurements = np.array([[0.05, 6.0]])
        config = SurvivalAwareAssociationConfig(
            detection_probability=0.95,
            visibility_probability=1.0,
            birth_weight=0.05,
            clutter_weight=0.05,
        )
        associator = build_survival_aware_linear_gaussian_hypothesis_associator(
            np.array([[1.0]]),
            np.array([[0.1]]),
            gates=NISGate(threshold=9.0),
            config=config,
        )

        result = associator(tracks, measurements, measurement_axis="columns")

        self.assertEqual(result.matches, [(0, 0)])
        self.assertEqual(result.unmatched_measurement_indices, [1])


if __name__ == "__main__":
    unittest.main()
