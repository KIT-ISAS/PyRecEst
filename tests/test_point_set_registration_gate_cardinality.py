import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import solve_gated_assignment


class TestGatedAssignmentCardinality(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_gate_boundary_still_maximizes_cardinality(self):
        assignment = solve_gated_assignment(
            array([[float("inf"), 1.0], [1.0, 0.0]]),
            max_cost=1.0,
        )

        npt.assert_array_equal(assignment, array([1, 0]))
