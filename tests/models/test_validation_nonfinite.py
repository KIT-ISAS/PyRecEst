import unittest

from pyrecest.backend import array
from pyrecest.models.validation import (
    validate_measurement_matrix,
    validate_measurement_vector,
    validate_state_vector,
    validate_transition_matrix,
)


class TestModelValidationNonfinite(unittest.TestCase):
    def test_validate_vectors_reject_nonfinite_values(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_state_vector(array([value]))
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_measurement_vector(array([value]))

    def test_validate_matrices_reject_nonfinite_values(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_transition_matrix(array([[value]]))
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_measurement_matrix(array([[value]]))


if __name__ == "__main__":
    unittest.main()
