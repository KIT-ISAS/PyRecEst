import unittest

import numpy as np
from pyrecest.models import continuous_to_discrete_lti


class TestContinuousToDiscreteLtiComplexValidation(unittest.TestCase):
    def test_rejects_complex_system_and_noise_matrices(self):
        invalid_cases = [
            (
                "continuous_matrix",
                {
                    "continuous_matrix": np.array([[1.0 + 2.0j]]),
                },
            ),
            (
                "noise_input_matrix",
                {
                    "continuous_matrix": np.zeros((1, 1)),
                    "noise_input_matrix": np.array([[1.0 + 2.0j]]),
                    "continuous_noise_covariance": np.ones((1, 1)),
                },
            ),
            (
                "continuous_noise_covariance",
                {
                    "continuous_matrix": np.zeros((1, 1)),
                    "noise_input_matrix": np.ones((1, 1)),
                    "continuous_noise_covariance": np.array([[1.0 + 2.0j]]),
                },
            ),
        ]

        for parameter_name, kwargs in invalid_cases:
            with self.subTest(parameter_name=parameter_name):
                with self.assertRaisesRegex(ValueError, parameter_name):
                    continuous_to_discrete_lti(**kwargs)


if __name__ == "__main__":
    unittest.main()
