import unittest

import numpy as np

from pyrecest.calibration.time_offset import (
    aggregate_time_offset_sweeps,
    apply_time_offset,
    interpolate_reference_values,
)


class TimeOffsetTemporalValidationTest(unittest.TestCase):
    def test_temporal_scalar_offsets_are_rejected(self):
        invalid_offsets = (
            np.datetime64("2020-01-01T00:00:00.000000001"),
            np.array(
                np.datetime64("2020-01-01T00:00:00.000000001"),
                dtype=object,
            ),
        )

        for offset_s in invalid_offsets:
            with self.subTest(offset_s=offset_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "offset_s must be a finite scalar",
                ):
                    apply_time_offset(np.array([0.0]), offset_s)

    def test_temporal_max_time_delta_is_rejected(self):
        invalid_deltas = (
            np.timedelta64(2, "ns"),
            np.array(np.timedelta64(2, "ns"), dtype=object),
        )

        for max_time_delta_s in invalid_deltas:
            with self.subTest(max_time_delta_s=max_time_delta_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_time_delta_s must be nonnegative",
                ):
                    interpolate_reference_values(
                        np.array([0.0, 1.0]),
                        np.array([[0.0], [1.0]]),
                        np.array([0.5]),
                        max_time_delta_s=max_time_delta_s,
                    )

    def test_temporal_object_arrays_are_rejected_as_times(self):
        times = np.array(
            [np.datetime64("2020-01-01T00:00:00.000000001")],
            dtype=object,
        )

        with self.assertRaisesRegex(
            ValueError,
            "times_s must contain real numeric values",
        ):
            apply_time_offset(times, 0.0)

    def test_temporal_summary_offsets_are_rejected(self):
        sweep = [
            [
                {
                    "time_offset_s": np.array(
                        np.datetime64("2020-01-01T00:00:00.000000001"),
                        dtype=object,
                    ),
                    "count": 1.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "rmse": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                }
            ]
        ]

        with self.assertRaisesRegex(
            ValueError,
            "time_offset_s must be a real scalar",
        ):
            aggregate_time_offset_sweeps(sweep)


if __name__ == "__main__":
    unittest.main()
