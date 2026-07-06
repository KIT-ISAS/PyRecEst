import numpy as np
import pytest

from pyrecest._backend.numpy import random


@pytest.mark.parametrize("population", [0, -1, np.array(0), np.array(-3)])
def test_choice_rejects_non_positive_integer_population_for_empty_sample(population):
    with pytest.raises(ValueError, match="positive integer"):
        random.choice(population, size=0)


@pytest.mark.parametrize(
    ("population", "axis"),
    [
        (np.empty((0,), dtype=int), 0),
        (np.empty((2, 0), dtype=int), 1),
    ],
)
def test_choice_rejects_empty_array_population_axis_for_empty_sample(population, axis):
    with pytest.raises(ValueError, match="non-empty array"):
        random.choice(population, size=0, axis=axis)


def test_choice_allows_empty_non_population_dimensions():
    random.seed(0)

    samples = random.choice(np.empty((2, 0), dtype=int), size=1, axis=0)

    assert samples.shape == (1, 0)
