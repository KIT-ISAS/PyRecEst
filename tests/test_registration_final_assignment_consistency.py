import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, zeros
from pyrecest.utils.point_set_registration import (
    AffineTransform,
    joint_registration_assignment,
)


class TestRegistrationFinalAssignmentConsistency(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_final_assignment_is_resolved_for_final_transform(self):
        reference = array([[0.0], [10.0]])
        moving = array([[4.0], [20.0]])

        result = joint_registration_assignment(
            reference,
            moving,
            model="translation",
            initial_transform=AffineTransform(eye(1), zeros(1)),
            max_cost=6.0,
            max_iterations=1,
        )

        npt.assert_array_equal(result.assignment, array([0, 1]))
        npt.assert_allclose(result.transform.offset, array([4.0]))
        npt.assert_allclose(
            result.transformed_reference_points,
            array([[4.0], [14.0]]),
        )
        npt.assert_allclose(result.matched_costs, array([0.0, 6.0]))
        self.assertFalse(result.converged)


if __name__ == "__main__":
    unittest.main()
