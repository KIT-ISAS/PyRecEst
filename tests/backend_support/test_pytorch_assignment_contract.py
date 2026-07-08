"""Regression tests for PyTorch backend assignment and logical helpers."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_assignment_accepts_numpy_style_advanced_indices():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

x = backend.zeros((3, 3))
assigned = backend.assignment(x, [5.0, 7.0], [(0, 1), (1, 2)])
summed = backend.assignment_by_sum(
    backend.ones((3, 3)),
    [5.0, 7.0],
    [(0, 1), (1, 2)],
)
singleton_assigned = backend.assignment(x, 9.0, [(2, 0)])
singleton_summed = backend.assignment_by_sum(
    backend.ones((3, 3)),
    4.0,
    [(2, 0)],
)
duplicate_sum = backend.assignment_by_sum(backend.zeros(3), [1.0, 2.0], [0, 0])

assert backend.to_numpy(assigned).tolist() == [
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 7.0],
    [0.0, 0.0, 0.0],
]
assert backend.to_numpy(summed).tolist() == [
    [1.0, 6.0, 1.0],
    [1.0, 1.0, 8.0],
    [1.0, 1.0, 1.0],
]
assert backend.to_numpy(singleton_assigned).tolist() == [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [9.0, 0.0, 0.0],
]
assert backend.to_numpy(singleton_summed).tolist() == [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [5.0, 1.0, 1.0],
]
assert backend.to_numpy(duplicate_sum).tolist() == [3.0, 0.0, 0.0]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_assignment_accepts_list_and_boolean_indices():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

vector = backend.zeros(3)
by_list = backend.assignment(vector, [4.0, 5.0], [0, 2])
by_slice = backend.assignment(vector, [1.0, 2.0], slice(1, 3))
by_slice_sum = backend.assignment_by_sum(vector, [1.0, 2.0], slice(1, 3))
by_array_like = backend.assignment(
    [[0.0, 0.0], [0.0, 0.0]],
    [4.0, 5.0],
    [(0, 1), (1, 0)],
)
by_array_like_sum = backend.assignment_by_sum(
    [0.0, 0.0, 0.0], [2.0, 3.0], [0, 2]
)
empty_array_like = backend.assignment([1.0, 2.0, 3.0], 99.0, [])
empty_sum_array_like = backend.assignment_by_sum([1.0, 2.0, 3.0], 99.0, [])

matrix = backend.zeros((3, 3))
by_mask = backend.assignment(
    matrix,
    [1.0, 2.0],
    [
        [True, False, False],
        [False, True, False],
        [False, False, False],
    ],
)

logical = backend.logical_and(backend.asarray([1, 0]), backend.asarray([2, 3]))

assert backend.to_numpy(by_list).tolist() == [4.0, 0.0, 5.0]
assert backend.to_numpy(by_slice).tolist() == [0.0, 1.0, 2.0]
assert backend.to_numpy(by_slice_sum).tolist() == [0.0, 1.0, 2.0]
assert backend.to_numpy(by_array_like).tolist() == [[0.0, 4.0], [5.0, 0.0]]
assert backend.to_numpy(by_array_like_sum).tolist() == [2.0, 0.0, 3.0]
assert backend.to_numpy(empty_array_like).tolist() == [1.0, 2.0, 3.0]
assert backend.to_numpy(empty_sum_array_like).tolist() == [1.0, 2.0, 3.0]
assert backend.to_numpy(by_mask).tolist() == [
    [1.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0],
]
assert backend.to_numpy(logical).tolist() == [True, False]
""",
    )

    assert result.returncode == 0, result.stderr
