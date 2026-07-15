import unittest

import numpy as np
from pyrecest.filters.discrete_state import (
    discrete_forward_backward,
    imm_forward_backward,
)


class TestDiscreteStatePriorNormalization(unittest.TestCase):
    def test_forward_backward_normalizes_large_finite_prior_without_overflow(self):
        maximum = np.finfo(float).max
        result = discrete_forward_backward(
            np.zeros((1, 2)),
            np.eye(2),
            initial_probabilities=np.array([maximum, maximum / 2.0]),
        )

        np.testing.assert_allclose(
            result.filtered_probabilities[0],
            np.array([2.0 / 3.0, 1.0 / 3.0]),
        )

    def test_imm_normalizes_large_finite_state_and_mode_priors(self):
        maximum = np.finfo(float).max
        result = imm_forward_backward(
            np.zeros((1, 2)),
            [np.eye(2), np.eye(2)],
            np.eye(2),
            initial_state_probabilities=np.array([maximum, maximum / 2.0]),
            initial_mode_probabilities=np.array([maximum, maximum / 2.0]),
        )

        expected = np.array([2.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(result.filtered_state_probabilities[0], expected)
        np.testing.assert_allclose(result.filtered_mode_probabilities[0], expected)


if __name__ == "__main__":
    unittest.main()
