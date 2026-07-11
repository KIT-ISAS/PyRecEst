import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


class _StateOnlyGaussianFilter:
    def __init__(self, mean):
        self.filter_state = GaussianDistribution(
            array([mean]), array([[1.0]]), check_validity=False
        )


def _filter_bank():
    return [_StateOnlyGaussianFilter(0.0), _StateOnlyGaussianFilter(1.0)]


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class IMMComplexInputValidationTest(unittest.TestCase):
    def test_rejects_complex_transition_matrix(self):
        transition_matrix = np.array(
            [[1.0 + 1.0j, 0.0], [0.0, 1.0]], dtype=np.complex128
        )

        with self.assertRaisesRegex(
            ValueError, "transition_matrix must contain real values"
        ):
            InteractingMultipleModelFilter(_filter_bank(), transition_matrix)

    def test_rejects_complex_mode_probabilities(self):
        mode_probabilities = np.array([0.5 + 0.5j, 0.5], dtype=np.complex128)

        with self.assertRaisesRegex(
            ValueError, "mode_probabilities must contain real values"
        ):
            InteractingMultipleModelFilter(
                _filter_bank(),
                transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
                mode_probabilities=mode_probabilities,
            )

    def test_rejects_complex_likelihoods(self):
        imm = InteractingMultipleModelFilter(
            _filter_bank(),
            transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
        )

        with self.assertRaisesRegex(ValueError, "likelihoods must contain real values"):
            imm.update_mode_probabilities(
                likelihoods=np.array([1.0 + 2.0j, 1.0], dtype=np.complex128)
            )

    def test_rejects_complex_log_likelihoods(self):
        imm = InteractingMultipleModelFilter(
            _filter_bank(),
            transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
        )

        with self.assertRaisesRegex(
            ValueError, "log_likelihoods must contain real values"
        ):
            imm.update_mode_probabilities(
                log_likelihoods=np.array([0.0 + 1.0j, -1.0], dtype=np.complex128)
            )

    def test_rejects_complex_moment_match_weights(self):
        states = [
            GaussianDistribution(array([0.0]), array([[1.0]])),
            GaussianDistribution(array([1.0]), array([[1.0]])),
        ]

        with self.assertRaisesRegex(ValueError, "weights must contain real values"):
            InteractingMultipleModelFilter._moment_match_gaussians(
                states, np.array([0.5 + 0.0j, 0.5], dtype=np.complex128)
            )


if __name__ == "__main__":
    unittest.main()
