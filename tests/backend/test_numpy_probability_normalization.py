import numpy as np

from pyrecest._backend.numpy import random


def test_choice_normalizes_large_finite_unnormalized_probabilities():
    weights = np.array([1e308, 1e308])

    random.seed(123)
    samples = random.choice(np.arange(2), size=16, p=weights)

    assert samples.shape == (16,)
    assert np.all((samples == 0) | (samples == 1))


def test_multinomial_normalizes_large_finite_unnormalized_probabilities():
    weights = np.array([1e308, 1e308])

    random.seed(123)
    samples = random.multinomial(5, weights, size=4)

    assert samples.shape == (4, 2)
    np.testing.assert_array_equal(samples.sum(axis=-1), np.full(4, 5))
