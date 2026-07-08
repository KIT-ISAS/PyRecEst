"""Regression tests for NumPy backend ``random.choice`` populations."""

from __future__ import annotations

import numpy as np
import pyrecest._backend.numpy.random as numpy_random
import pytest


@pytest.mark.parametrize(
    "population",
    (
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.array(True),
        np.array(False),
        np.array(True, dtype=object),
        np.array(False, dtype=object),
    ),
)
@pytest.mark.parametrize("size", (None, (0,), (1,)))
def test_choice_rejects_boolean_scalar_population(population, size):
    with pytest.raises(ValueError, match="positive integer or a non-empty array"):
        numpy_random.choice(population, size=size)


@pytest.mark.parametrize(
    "population",
    (
        np.timedelta64(2, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
        np.array(np.timedelta64(2, "ns")),
        np.array(np.datetime64("1970-01-01T00:00:00.000000002", "ns")),
    ),
)
@pytest.mark.parametrize("size", (None, (0,), (1,)))
def test_choice_rejects_temporal_scalar_population(population, size):
    with pytest.raises(ValueError, match="positive integer or a non-empty array"):
        numpy_random.choice(population, size=size)


def test_choice_keeps_boolean_array_populations_valid():
    selected = numpy_random.choice(np.array([True, False]), size=(4,), replace=True)

    assert selected.shape == (4,)
    assert selected.dtype == np.bool_


def test_choice_keeps_temporal_array_populations_valid():
    population = np.array(
        [
            "2026-07-09T00:00:00.000000001",
            "2026-07-09T00:00:00.000000002",
        ],
        dtype="datetime64[ns]",
    )

    selected = numpy_random.choice(population, size=(4,), replace=True)

    assert selected.shape == (4,)
    assert selected.dtype.kind == "M"
