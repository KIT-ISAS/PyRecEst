import pytest

from tests.support.backend_runner import run_backend_code


def test_autograd_vmap_accepts_array_like_inputs():
    pytest.importorskip("autograd")

    code = """
import pyrecest.backend as backend


def add_one(row):
    return row + 1

result = backend.vmap(add_one)([[1, 2], [3, 4]])
assert backend.to_numpy(result).tolist() == [[2, 3], [4, 5]]
"""
    result = run_backend_code("autograd", code)
    assert result.returncode == 0, result.stderr
