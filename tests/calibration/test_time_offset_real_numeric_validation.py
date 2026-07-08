import unittest

import numpy as np

from pyrecest.calibration import (
    apply_time_offset,
    interpolate_reference_values,
    make_offset_grid,
)


class TimeOffsetRealNumericValidationTest(unittest.TestCase):
    def test_apply_time_offset_rejects_numpy_complex_scalar_inside_object_array(self):
        times = np.array([np.complex128(complex(1.0, 2.0))], dtype=object)

        with self.assertRaisesRegex(
            ValueError,
            "times_s must contain real numeric values",
        ):
            apply_time_offset(times, 0.0)

    def test_make_offset_grid_rejects_numpy_complex_object_scalar(self):
        min_s = np.array(np.complex128(complex(0.0, 1.0)), dtype=object)

        with self.assertRaisesRegex(ValueError, "min_s must be a finite scalar"):
            make_offset_grid(min_s, 1.0, 1.0)

    def test_apply_time_offset_rejects_datetime_like_object_arrays(self):
        for times in (
            np.array([np.datetime64("2020-01-01")], dtype=object),
            np.array([np.timedelta64(2, "ns")], dtype=object),
        ):
            with self.subTest(times=times):
                with self.assertRaisesRegex(
                    ValueError,
                    "times_s must contain real numeric values",
                ):
                    apply_time_offset(times, 0.0)

    def test_apply_time_offset_rejects_temporal_scalar_offsets(self):
        times = np.array([0.0, 1.0])

        for offset_s in (
            np.timedelta64(1, "ns"),
            np.array(np.timedelta64(1, "ns"), dtype=object),
        ):
            with self.subTest(offset_s=offset_s):
                with self.assertRaisesRegex(ValueError, "offset_s must be a finite scalar"):
                    apply_time_offset(times, offset_s)

    def test_interpolation_rejects_temporal_max_time_delta_scalar(self):
        reference_times = np.array([0.0, 1.0])
        reference_values = np.array([[0.0], [1.0]])
        query_times = np.array([0.5])

        with self.assertRaisesRegex(ValueError, "max_time_delta_s must be nonnegative"):
            interpolate_reference_values(
                reference_times,
                reference_values,
                query_times,
                max_time_delta_s=np.timedelta64(1, "ns"),
            )


if __name__ == "__main__":
    unittest.main()
