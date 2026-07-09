"""Regression tests for temporal scalar validation in track completion."""

from __future__ import annotations

import unittest

import numpy as np
from pyrecest.utils.track_completion import (
    CompletionCandidate,
    enumerate_fragment_completion_paths,
)


def _provider(session: int, observation: int, target_session: int):
    del session, observation, target_session
    return [1]


class TestTrackCompletionTemporalValidation(unittest.TestCase):
    def test_temporal_path_limits_are_not_interpreted_as_integer_payloads(self):
        tracks = [[0, None]]

        with self.assertRaisesRegex(ValueError, "max_path_length"):
            enumerate_fragment_completion_paths(
                tracks,
                direction="suffix",
                max_path_length=np.timedelta64(1, "ns"),
                candidate_provider=_provider,
            )

        with self.assertRaisesRegex(ValueError, "max_paths_per_fragment"):
            enumerate_fragment_completion_paths(
                tracks,
                direction="suffix",
                max_paths_per_fragment=np.datetime64(
                    "1970-01-01T00:00:00.000000001",
                ),
                candidate_provider=_provider,
            )

    def test_temporal_candidate_observations_are_rejected(self):
        tracks = [[0, None]]
        invalid_candidates = (
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.array(np.timedelta64(1, "ns")),
            np.array(np.datetime64("1970-01-01T00:00:00.000000001")),
            np.array(np.timedelta64(1, "ns"), dtype=object),
            CompletionCandidate(np.datetime64("1970-01-01T00:00:00.000000001")),
        )

        for invalid_candidate in invalid_candidates:
            with self.subTest(invalid_candidate=repr(invalid_candidate)):

                def provider(session: int, observation: int, target_session: int):
                    del session, observation, target_session
                    return [invalid_candidate]

                with self.assertRaisesRegex(
                    ValueError,
                    "candidate observations must be non-negative integers",
                ):
                    enumerate_fragment_completion_paths(
                        tracks,
                        direction="suffix",
                        candidate_provider=provider,
                    )

    def test_temporal_candidate_sessions_are_ignored_not_unwrapped(self):
        tracks = [[0, None, None]]

        def candidate_session_provider(
            session: int,
            observation: int,
            direction: str,
        ):
            del session, observation, direction
            return [np.timedelta64(1, "ns")]

        paths = enumerate_fragment_completion_paths(
            tracks,
            direction="suffix",
            candidate_provider=_provider,
            candidate_session_provider=candidate_session_provider,
        )

        self.assertEqual(paths, [])

    def test_temporal_candidate_scores_are_rejected(self):
        tracks = [[0, None]]
        invalid_scores = (
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.array(np.timedelta64(1, "ns"), dtype=object),
        )

        for invalid_score in invalid_scores:
            with self.subTest(invalid_score=repr(invalid_score)):

                def provider(session: int, observation: int, target_session: int):
                    del session, observation, target_session
                    return [CompletionCandidate(1, score=invalid_score)]

                with self.assertRaisesRegex(
                    ValueError,
                    "candidate scores must be a finite real scalar",
                ):
                    enumerate_fragment_completion_paths(
                        tracks,
                        direction="suffix",
                        candidate_provider=provider,
                    )

    def test_temporal_path_scores_are_rejected(self):
        tracks = [[0, None]]

        with self.assertRaisesRegex(
            ValueError,
            "path scores must be a finite real scalar",
        ):
            enumerate_fragment_completion_paths(
                tracks,
                direction="suffix",
                candidate_provider=_provider,
                score_path=lambda steps: np.timedelta64(len(steps), "ns"),
            )


if __name__ == "__main__":
    unittest.main()
