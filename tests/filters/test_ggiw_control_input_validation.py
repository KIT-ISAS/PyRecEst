import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, diag, eye
from pyrecest.filters import GGIWTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="GGIW tracker tests currently use numpy.testing assertions",
)
class GGIWControlInputValidationTest(unittest.TestCase):
    @staticmethod
    def _tracker():
        return GGIWTracker(
            kinematic_state=array([0.0, 0.0, 1.0, -1.0]),
            covariance=diag(array([1.0, 1.0, 0.25, 0.25])),
            extent=diag(array([4.0, 1.0])),
            extent_degrees_of_freedom=12.0,
            gamma_shape=4.0,
            gamma_rate=2.0,
            measurement_matrix=array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ),
        )

    def test_rejects_broadcastable_wrong_sized_input_without_mutating_state(self):
        tracker = self._tracker()
        prior_state = tracker.kinematic_state.copy()
        prior_covariance = tracker.covariance.copy()
        prior_extent_scale = tracker.extent_scale.copy()
        prior_degrees_of_freedom = tracker.extent_degrees_of_freedom
        prior_gamma_shape = tracker.gamma_shape
        prior_gamma_rate = tracker.gamma_rate

        with self.assertRaisesRegex(ValueError, r"inputs must have shape \(4,\)"):
            tracker.predict_linear(
                system_matrix=eye(4),
                sys_noise=eye(4),
                inputs=array([2.0]),
            )

        npt.assert_array_equal(tracker.kinematic_state, prior_state)
        npt.assert_array_equal(tracker.covariance, prior_covariance)
        npt.assert_array_equal(tracker.extent_scale, prior_extent_scale)
        self.assertEqual(
            tracker.extent_degrees_of_freedom,
            prior_degrees_of_freedom,
        )
        self.assertEqual(tracker.gamma_shape, prior_gamma_shape)
        self.assertEqual(tracker.gamma_rate, prior_gamma_rate)


if __name__ == "__main__":
    unittest.main()
