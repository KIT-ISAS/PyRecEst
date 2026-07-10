"""Regression tests for temporal track-metric controls."""

import unittest

import numpy as np
from pyrecest.utils.track_metrics import (
    score_false_tracks,
    score_missed_tracks,
    track_latencies,
)


class TestTrackMetricsTemporalControls(unittest.TestCase):
    def test_track_latency_rejects_native_temporal_session_arrays(self):
        invalid_session_times = (
            np.array([1, 2], dtype="datetime64[ns]"),
            np.array([1, 2], dtype="timedelta64[ns]"),
        )

        for session_times in invalid_session_times:
            with self.subTest(dtype=str(session_times.dtype)):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_times must contain only finite numeric values",
                ):
                    track_latencies(
                        [[0, 0]],
                        [[0, 0]],
                        session_times=session_times,
                    )

    def test_track_latency_rejects_temporal_missed_values(self):
        invalid_missed_values = (
            np.datetime64(1, "ns"),
            np.timedelta64(1, "ns"),
            np.asarray(np.datetime64(1, "ns")),
            np.asarray(np.timedelta64(1, "ns")),
            np.array(np.datetime64(1, "ns"), dtype=object),
            np.array(np.timedelta64(1, "ns"), dtype=object),
        )

        for missed_value in invalid_missed_values:
            with self.subTest(missed_value=repr(missed_value)):
                with self.assertRaisesRegex(ValueError, "missed_value"):
                    track_latencies([[None]], [[0]], missed_value=missed_value)

    def test_track_metrics_reject_temporal_min_lengths(self):
        invalid_min_lengths = (
            np.datetime64(1, "ns"),
            np.timedelta64(1, "ns"),
            np.asarray(np.datetime64(1, "ns")),
            np.asarray(np.timedelta64(1, "ns")),
        )

        for metric in (score_false_tracks, score_missed_tracks):
            for min_length in invalid_min_lengths:
                with self.subTest(
                    metric=metric.__name__,
                    min_length=repr(min_length),
                ):
                    with self.assertRaisesRegex(ValueError, "min_length"):
                        metric([[0]], [[0]], min_length=min_length)


if __name__ == "__main__":
    unittest.main()
