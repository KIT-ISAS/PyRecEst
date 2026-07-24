"""Regression tests for strict reproducibility seed type validation."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from pyrecest.reproducibility import seed_all


@pytest.mark.parametrize(
    "seed",
    [
        1.0,
        np.float64(1.0),
        Decimal("1"),
        Fraction(1, 1),
    ],
)
def test_seed_all_rejects_noninteger_scalar_types(seed):
    with pytest.raises(ValueError, match="non-negative integer"):
        seed_all(seed)


@pytest.mark.parametrize("seed", [0, 1, np.int32(2), np.int64(3)])
def test_seed_all_accepts_integer_scalar_types(seed):
    assert seed_all(seed) == int(seed)
