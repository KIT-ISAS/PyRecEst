import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_dot_matches_numpy_matrix_and_vector_contracts():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

assert backend.__backend_name__ == "pytorch"

matrix_left = backend.array([[1.0, 2.0], [3.0, 4.0]])
matrix_right = backend.array([[5.0, 6.0], [7.0, 8.0]])
matrix_result = backend.dot(matrix_left, matrix_right)
assert matrix_result.shape == (2, 2)
assert backend.to_numpy(matrix_result).tolist() == [[19.0, 22.0], [43.0, 50.0]]

vector_matrix_result = backend.dot(
    backend.array([1.0, 2.0]),
    backend.array([[3.0, 4.0], [5.0, 6.0]]),
)
assert vector_matrix_result.shape == (2,)
assert backend.to_numpy(vector_matrix_result).tolist() == [13.0, 16.0]

batched_result = backend.dot(backend.array([[[1.0, 2.0], [3.0, 4.0]]]), matrix_right)
assert batched_result.shape == (1, 2, 2)
assert backend.to_numpy(batched_result).tolist() == [[[19.0, 22.0], [43.0, 50.0]]]
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
