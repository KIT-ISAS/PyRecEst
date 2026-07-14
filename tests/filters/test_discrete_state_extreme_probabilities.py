import numpy as np

from pyrecest.filters.discrete_state import (
    discrete_forward_backward,
    imm_forward_backward,
)


def test_forward_backward_normalizes_extreme_finite_initial_probabilities():
    max_finite = np.finfo(float).max

    result = discrete_forward_backward(
        np.zeros((1, 2)),
        np.eye(2),
        initial_probabilities=np.array([max_finite, max_finite / 2.0]),
    )

    expected = np.array([2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(result.filtered_probabilities[0], expected)
    np.testing.assert_allclose(result.smoothed_probabilities[0], expected)
    assert np.isfinite(result.log_marginal_likelihood)


def test_imm_normalizes_extreme_finite_state_and_mode_probabilities():
    max_finite = np.finfo(float).max

    result = imm_forward_backward(
        np.zeros((1, 2)),
        [np.eye(2), np.eye(2)],
        np.eye(2),
        initial_state_probabilities=np.array([max_finite, max_finite / 2.0]),
        initial_mode_probabilities=np.array([max_finite / 2.0, max_finite]),
    )

    expected_state = np.array([2.0 / 3.0, 1.0 / 3.0])
    expected_mode = np.array([1.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(result.filtered_state_probabilities[0], expected_state)
    np.testing.assert_allclose(result.filtered_mode_probabilities[0], expected_mode)
    np.testing.assert_allclose(result.filtered_joint_probabilities.sum(), 1.0)
