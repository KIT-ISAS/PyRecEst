import unittest

import numpy as np
from pyrecest.tracking.event_records import TrackingEvent, TrackingRecord


def _temporal_scalars():
    return (
        np.datetime64("1970-01-01T00:00:00.000000001"),
        np.timedelta64(1, "ns"),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001"), dtype=object),
        np.array(np.timedelta64(1, "ns"), dtype=object),
    )


class TestTrackingEventScalarValidation(unittest.TestCase):
    def test_time_accepts_scalar_numeric_values(self):
        event = TrackingEvent(time=np.array(1.25), source="sensor")

        self.assertEqual(event.time, 1.25)

    def test_time_rejects_bool_text_and_non_scalar_values(self):
        invalid_times = (
            True,
            np.bool_(False),
            "1.0",
            b"1.0",
            np.array("1.0", dtype=object),
            np.array([1.0]),
            np.array([[1.0]]),
        )
        for time in invalid_times:
            with self.subTest(time=time):
                with self.assertRaisesRegex(ValueError, "time must be finite"):
                    TrackingEvent(time=time, source="sensor")

    def test_time_rejects_temporal_scalars(self):
        for time in _temporal_scalars():
            with self.subTest(time=repr(time)):
                with self.assertRaisesRegex(ValueError, "time must be finite"):
                    TrackingEvent(time=time, source="sensor")

    def test_measurement_and_covariance_reject_object_temporal_values(self):
        temporal_measurement = np.array([np.timedelta64(1, "ns")], dtype=object)
        temporal_covariance = np.array([[np.datetime64("1970-01-01")]], dtype=object)

        with self.assertRaisesRegex(ValueError, "measurement must contain"):
            TrackingEvent(time=0.0, source="sensor", measurement=temporal_measurement)
        with self.assertRaisesRegex(ValueError, "covariance must contain"):
            TrackingEvent(time=0.0, source="sensor", covariance=temporal_covariance)


class TestTrackingRecordScalarValidation(unittest.TestCase):
    def _valid_record_kwargs(self):
        return {
            "time": 0.0,
            "source": "sensor",
            "action": "update",
            "prior_mean": np.array([0.0, 0.0]),
            "prior_cov": np.eye(2),
            "posterior_mean": np.array([1.0, 0.0]),
            "posterior_cov": np.eye(2),
        }

    def test_nis_accepts_nonnegative_scalar_numeric_values(self):
        record = TrackingRecord(**self._valid_record_kwargs(), nis=np.array(2.5))

        self.assertEqual(record.nis, 2.5)

    def test_time_rejects_bool_text_and_non_scalar_values(self):
        invalid_times = (
            True,
            np.bool_(False),
            "1.0",
            b"1.0",
            np.array("1.0", dtype=object),
            np.array([1.0]),
            np.array([[1.0]]),
        )
        for time in invalid_times:
            kwargs = self._valid_record_kwargs()
            kwargs["time"] = time
            with self.subTest(time=time):
                with self.assertRaisesRegex(ValueError, "time must be finite"):
                    TrackingRecord(**kwargs)

    def test_time_rejects_temporal_scalars(self):
        for time in _temporal_scalars():
            kwargs = self._valid_record_kwargs()
            kwargs["time"] = time
            with self.subTest(time=repr(time)):
                with self.assertRaisesRegex(ValueError, "time must be finite"):
                    TrackingRecord(**kwargs)

    def test_nis_rejects_bool_text_non_scalar_and_negative_values(self):
        invalid_nis_values = (
            True,
            np.bool_(False),
            "2.5",
            b"2.5",
            np.array("2.5", dtype=object),
            np.array([1.0]),
            -1.0,
            np.nan,
        )

        for nis in invalid_nis_values:
            with self.subTest(nis=nis):
                with self.assertRaisesRegex(
                    ValueError,
                    "nis must be finite and nonnegative",
                ):
                    TrackingRecord(**self._valid_record_kwargs(), nis=nis)

    def test_nis_rejects_temporal_scalars(self):
        for nis in _temporal_scalars():
            with self.subTest(nis=repr(nis)):
                with self.assertRaisesRegex(
                    ValueError,
                    "nis must be finite and nonnegative",
                ):
                    TrackingRecord(**self._valid_record_kwargs(), nis=nis)

    def test_state_and_covariance_reject_object_temporal_values(self):
        temporal_vector = np.array([np.timedelta64(1, "ns")], dtype=object)
        temporal_matrix = np.array([[np.datetime64("1970-01-01")]], dtype=object)

        kwargs = self._valid_record_kwargs()
        kwargs["prior_mean"] = temporal_vector
        with self.assertRaisesRegex(ValueError, "prior_mean must contain"):
            TrackingRecord(**kwargs)

        kwargs = self._valid_record_kwargs()
        kwargs["prior_cov"] = temporal_matrix
        with self.assertRaisesRegex(ValueError, "prior_cov must contain"):
            TrackingRecord(**kwargs)


if __name__ == "__main__":
    unittest.main()
