import unittest

import numpy as np
import pyrecest.backend
from pyrecest.utils import (
    min_cost_max_cardinality_assignment,
    murty_k_best_assignments,
)


class AssignmentCostMatrixValidationTest(unittest.TestCase):
    @staticmethod
    def _solvers():
        return (
            ("murty", lambda matrix: murty_k_best_assignments(matrix, k=1)),
            ("max_cardinality", min_cost_max_cardinality_assignment),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_nan_cost_matrix_entries_are_rejected(self):
        for solver_name, solver in self._solvers():
            with self.subTest(solver=solver_name):
                with self.assertRaisesRegex(ValueError, "positive infinity"):
                    solver(np.array([[np.nan, 1.0]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_negative_infinite_cost_matrix_entries_are_rejected(self):
        for solver_name, solver in self._solvers():
            with self.subTest(solver=solver_name):
                with self.assertRaisesRegex(ValueError, "positive infinity"):
                    solver(np.array([[-np.inf, 1.0]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_boolean_cost_matrix_is_rejected(self):
        for solver_name, solver in self._solvers():
            with self.subTest(solver=solver_name):
                with self.assertRaisesRegex(ValueError, "not boolean"):
                    solver(np.array([[True, False]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_object_boolean_cost_matrix_entries_are_rejected(self):
        matrices = (
            np.array([[1.0, True]], dtype=object),
            np.array([[False, 2.0]], dtype=object),
        )
        for solver_name, solver in self._solvers():
            for matrix in matrices:
                with self.subTest(solver=solver_name, matrix=matrix.tolist()):
                    with self.assertRaisesRegex(ValueError, "not boolean"):
                        solver(matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_text_cost_matrix_entries_are_rejected(self):
        text_matrices = (
            np.array([["1.0", "2.0"]]),
            np.array([[b"1.0", b"2.0"]], dtype=object),
        )
        for solver_name, solver in self._solvers():
            for matrix in text_matrices:
                with self.subTest(solver=solver_name, dtype=str(matrix.dtype)):
                    with self.assertRaisesRegex(
                        ValueError, "cost_matrix must be numeric"
                    ):
                        solver(matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_temporal_cost_matrix_entries_are_rejected(self):
        temporal_matrices = (
            np.array([[np.timedelta64(1, "ns")]]),
            np.array([[np.datetime64("1970-01-01T00:00:00.000000001")]]),
            np.array([[np.timedelta64(1, "ns")]], dtype=object),
            np.array(
                [[np.datetime64("1970-01-01T00:00:00.000000001")]],
                dtype=object,
            ),
        )
        for solver_name, solver in self._solvers():
            for matrix in temporal_matrices:
                with self.subTest(solver=solver_name, dtype=str(matrix.dtype)):
                    with self.assertRaisesRegex(
                        ValueError, "cost_matrix must be numeric"
                    ):
                        solver(matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_text_non_assignment_costs_are_rejected(self):
        matrix = np.array([[1.0]])
        invalid_costs = (
            {"row_non_assignment_costs": np.array(["0.5"])},
            {"col_non_assignment_costs": np.array([b"0.5"], dtype=object)},
        )
        for kwargs in invalid_costs:
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaisesRegex(ValueError, "must be numeric and finite"):
                    murty_k_best_assignments(matrix, k=1, **kwargs)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_temporal_non_assignment_costs_are_rejected(self):
        matrix = np.array([[1.0]])
        invalid_costs = (
            {"row_non_assignment_costs": np.array([np.timedelta64(1, "ns")])},
            {
                "col_non_assignment_costs": np.array(
                    [np.datetime64("1970-01-01T00:00:00.000000001")],
                    dtype=object,
                )
            },
        )
        for kwargs in invalid_costs:
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaisesRegex(ValueError, "must be numeric and finite"):
                    murty_k_best_assignments(matrix, k=1, **kwargs)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_object_boolean_non_assignment_costs_are_rejected(self):
        matrix = np.array([[1.0]])
        invalid_costs = (
            {"row_non_assignment_costs": np.array([True], dtype=object)},
            {"col_non_assignment_costs": np.array([False], dtype=object)},
        )
        for kwargs in invalid_costs:
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaisesRegex(ValueError, "must be numeric and finite"):
                    murty_k_best_assignments(matrix, k=1, **kwargs)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_positive_infinity_remains_infeasible_edge_sentinel(self):
        solutions = murty_k_best_assignments(
            np.array([[np.inf, 1.0]]),
            k=1,
            row_non_assignment_costs=5.0,
            col_non_assignment_costs=0.0,
        )

        self.assertEqual(len(solutions), 1)
        np.testing.assert_array_equal(solutions[0]["assignment"], np.array([1]))
        self.assertAlmostEqual(solutions[0]["cost"], 1.0)


if __name__ == "__main__":
    unittest.main()
