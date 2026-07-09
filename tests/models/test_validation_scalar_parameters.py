"""Regression tests for model-validation scalar parameters."""

from __future__ import annotations

import unittest

import numpy as np
from pyrecest.models.validation import (
    infer_state_dim_from_distribution,
    validate_covariance_matrix,
    validate_matrix,
    validate_state_vector,
)


class TestModelValidationScalarParameters(unittest.TestCase):
    def test_symmetry_tolerances_reject_string_like_scalars(self) -> None:
        covariance = [[1.0, 0.0], [0.0, 1.0]]
        invalid_values = (
            "0.0",
            b"0.0",
            np.array("0.0"),
            np.array(b"0.0", dtype="S3"),
            np.array("0.0", dtype=object),
        )

        for tolerance_name in ("symmetric_rtol", "symmetric_atol"):
            for invalid_value in invalid_values:
                with self.subTest(
                    tolerance_name=tolerance_name,
                    invalid_value=repr(invalid_value),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{tolerance_name} must be a finite nonnegative scalar",
                    ):
                        validate_covariance_matrix(
                            covariance,
                            check_symmetric=True,
                            **{tolerance_name: invalid_value},
                        )

    def test_symmetry_tolerances_reject_temporal_scalars(self) -> None:
        covariance = [[1.0, 0.0], [0.0, 1.0]]
        invalid_values = (
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.array(np.timedelta64(1, "ns"), dtype=object),
            np.array(
                np.datetime64("1970-01-01T00:00:00.000000001"),
                dtype=object,
            ),
        )

        for tolerance_name in ("symmetric_rtol", "symmetric_atol"):
            for invalid_value in invalid_values:
                with self.subTest(
                    tolerance_name=tolerance_name,
                    invalid_value=repr(invalid_value),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{tolerance_name} must be a finite nonnegative scalar",
                    ):
                        validate_covariance_matrix(
                            covariance,
                            check_symmetric=True,
                            **{tolerance_name: invalid_value},
                        )

    def test_expected_dimensions_reject_temporal_scalars(self) -> None:
        vector = [1.0, 2.0]
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        invalid_values = (
            np.timedelta64(2, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000002"),
            np.array(np.timedelta64(2, "ns"), dtype=object),
            np.array(
                np.datetime64("1970-01-01T00:00:00.000000002"),
                dtype=object,
            ),
        )

        for invalid_value in invalid_values:
            with self.subTest(kind="dim", invalid_value=repr(invalid_value)):
                with self.assertRaisesRegex(
                    TypeError,
                    "dim must be an integer or None",
                ):
                    validate_state_vector(vector, state_dim=invalid_value)

            with self.subTest(kind="rows", invalid_value=repr(invalid_value)):
                with self.assertRaisesRegex(
                    TypeError,
                    "rows must be an integer or None",
                ):
                    validate_matrix(matrix, rows=invalid_value)

            with self.subTest(kind="cols", invalid_value=repr(invalid_value)):
                with self.assertRaisesRegex(
                    TypeError,
                    "cols must be an integer or None",
                ):
                    validate_matrix(matrix, cols=invalid_value)

    def test_temporal_vector_and_matrix_entries_are_rejected(self) -> None:
        temporal_vectors = (
            np.array(
                [
                    np.datetime64("2020-01-01T00:00:00.000000001"),
                    np.datetime64("2020-01-01T00:00:00.000000002"),
                ]
            ),
            np.array([np.timedelta64(1, "ns"), np.timedelta64(2, "ns")]),
        )
        temporal_matrices = (
            np.array(
                [
                    [
                        np.datetime64("2020-01-01T00:00:00.000000001"),
                        np.datetime64("2020-01-01T00:00:00.000000002"),
                    ],
                    [
                        np.datetime64("2020-01-01T00:00:00.000000003"),
                        np.datetime64("2020-01-01T00:00:00.000000004"),
                    ],
                ]
            ),
            np.array(
                [
                    [np.timedelta64(1, "ns"), np.timedelta64(2, "ns")],
                    [np.timedelta64(3, "ns"), np.timedelta64(4, "ns")],
                ]
            ),
        )

        for temporal_vector in temporal_vectors:
            with self.subTest(value=repr(temporal_vector)):
                with self.assertRaisesRegex(
                    ValueError,
                    "state must contain numeric non-boolean values",
                ):
                    validate_state_vector(temporal_vector)

        for temporal_matrix in temporal_matrices:
            with self.subTest(value=repr(temporal_matrix)):
                with self.assertRaisesRegex(
                    ValueError,
                    "matrix must contain numeric non-boolean values",
                ):
                    validate_matrix(temporal_matrix)

    def test_infer_state_dim_ignores_temporal_dim_attributes(self) -> None:
        class TemporalDimOnly:
            dim = np.timedelta64(2, "ns")

        with self.assertRaisesRegex(
            ValueError,
            "Could not infer a positive state dimension",
        ):
            infer_state_dim_from_distribution(TemporalDimOnly())

    def test_symmetry_tolerances_accept_numeric_scalar_arrays(self) -> None:
        covariance = [[1.0, 0.0], [0.0, 1.0]]

        validated = validate_covariance_matrix(
            covariance,
            check_symmetric=True,
            symmetric_rtol=np.array(0.0),
            symmetric_atol=np.array(0.0),
        )

        self.assertEqual(tuple(validated.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
