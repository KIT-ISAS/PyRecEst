import unittest

import numpy as np
from pyrecest.filters import OnlineTimeOffsetEstimator


def _nested_object_array(value, *, scalar=False):
    result = np.empty((), dtype=object) if scalar else np.empty((1,), dtype=object)
    if scalar:
        result[()] = np.asarray(value)
    else:
        result[0] = np.asarray(value)
    return result


class OnlineTimeOffsetEstimatorNestedInputTest(unittest.TestCase):
    def test_update_rejects_nested_nonreal_values_without_state_change(self):
        invalid_updates = (
            (
                {"residual": _nested_object_array(1.0 + 2.0j)},
                "residual must be real-valued numeric",
            ),
            (
                {"residual": _nested_object_array(True)},
                "residual must be real-valued numeric",
            ),
            (
                {"velocity": _nested_object_array("2.0")},
                "velocity must be real-valued numeric",
            ),
            (
                {"velocity": _nested_object_array(np.datetime64("2026-07-24"))},
                "velocity must be real-valued numeric",
            ),
            (
                {"measurement_variance": _nested_object_array(True, scalar=True)},
                "measurement_variance must be a finite scalar",
            ),
        )

        for override, message in invalid_updates:
            estimator = OnlineTimeOffsetEstimator(offset=1.0, variance=2.0)
            kwargs = {
                "residual": np.array([1.0]),
                "velocity": np.array([2.0]),
                "measurement_variance": 1.0,
            }
            kwargs.update(override)
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    estimator.update_from_position_residual(**kwargs)
                self.assertEqual(estimator.offset, 1.0)
                self.assertEqual(estimator.variance, 2.0)

    def test_update_preserves_nested_real_numeric_inputs(self):
        estimator = OnlineTimeOffsetEstimator(offset=0.0, variance=1.0)

        nis = estimator.update_from_position_residual(
            residual=_nested_object_array(10.0),
            velocity=_nested_object_array(5.0),
            measurement_variance=_nested_object_array(1.0, scalar=True),
        )

        self.assertTrue(np.isfinite(nis))
        self.assertGreater(estimator.offset, 0.0)
        self.assertLess(estimator.offset, 2.0)


if __name__ == "__main__":
    unittest.main()
