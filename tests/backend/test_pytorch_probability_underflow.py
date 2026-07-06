import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


def test_choice_normalizes_tiny_float64_probabilities_without_underflow():
    random.seed(0)
    probabilities = np.array([1.0e-300, 1.0e-300], dtype=np.float64)

    samples = random.choice(2, size=8, p=probabilities)

    assert samples.shape == (8,)
    assert torch.all((samples == 0) | (samples == 1))


def test_multinomial_normalizes_tiny_float64_probabilities_without_underflow():
    random.seed(0)
    probabilities = np.array([1.0e-300, 1.0e-300], dtype=np.float64)

    counts = random.multinomial(5, probabilities, size=3)

    assert counts.shape == (3, 2)
    assert torch.all(counts.sum(dim=-1) == 5)
