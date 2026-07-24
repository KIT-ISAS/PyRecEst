"""Regression tests for complex LTI discretization inputs."""

import unittest

import numpy as np
from pyrecest.models import continuous_to_discrete_lti


class TestContinuousToDiscreteLtiComplexValidation(unittest.TestCase):
    def test_rejects_complex_system_and_noise_matrices(self):
        nested_complex = np.empty((1, 1), dtype=object)
        nested_complex[0, 0] = np.array(1.0 + 2.0j)
        complex_variants = (
            np.array([[1.0 + 2.0j]]),
            np.array([[1.0 + 2.0j]], dtype=object),
            nested_complex,
        )
        parameter_cases = (
            (
                "continuous_matrix",
                lambda matrix: {
                    "continuous_matrix": matrix,
                },
            ),
            (
                "noise_input_matrix",
                lambda matrix: {
                    "continuous_matrix": np.zeros((1, 1)),
                    "noise_input_matrix": matrix,
                    "continuous_noise_covariance": np.ones((1, 1)),
                },
            ),
            (
                "continuous_noise_covariance",
                lambda matrix: {
                    "continuous_matrix": np.zeros((1, 1)),
                    "noise_input_matrix": np.ones((1, 1)),
                    "continuous_noise_covariance": matrix,
                },
            ),
        )

        for complex_matrix in complex_variants:
            for parameter_name, build_kwargs in parameter_cases:
                with self.subTest(
                    parameter_name=parameter_name,
                    dtype=complex_matrix.dtype,
                ):
                    with self.assertRaisesRegex(ValueError, parameter_name):
                        continuous_to_discrete_lti(**build_kwargs(complex_matrix))


if __name__ == "__main__":
    unittest.main()
