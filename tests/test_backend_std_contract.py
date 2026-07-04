import importlib.util

import pyrecest.backend as backend
import pytest
from tests.support.backend_runner import run_backend_code


def test_std_accepts_array_like_inputs_on_active_backend():
    result = backend.std([[1, 2, 3], [4, 5, 6]], axis=0, ddof=1, keepdims=True)

    assert tuple(result.shape) == (1, 3)
    assert backend.allclose(
        result,
        backend.array([[2.1213203435596424, 2.1213203435596424, 2.1213203435596424]]),
    )


@pytest.mark.backend_portable
def test_pytorch_std_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

result = backend.std([[1, 2, 3], [4, 5, 6]], axis=0, ddof=1, keepdims=True)
expected = backend.array([[2.1213203435596424, 2.1213203435596424, 2.1213203435596424]])
assert tuple(result.shape) == (1, 3)
assert result.dtype == backend.float64
assert backend.allclose(result, expected)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_jax_std_accepts_out_argument():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend
import pyrecest._backend.jax as raw_jax

values = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
expected = backend.array([[2.1213203435596424, 2.1213203435596424, 2.1213203435596424]])

public_out = backend.zeros((1, 3), dtype=backend.float64)
public_result = backend.std(
    values,
    axis=0,
    dtype=backend.float64,
    out=public_out,
    ddof=1,
    keepdims=True,
)
assert tuple(public_result.shape) == (1, 3)
assert backend.allclose(public_result, expected)

correction_result = backend.std(
    values,
    axis=0,
    dtype=backend.float64,
    correction=1,
    keepdims=True,
)
assert tuple(correction_result.shape) == (1, 3)
assert backend.allclose(correction_result, expected)

raw_out = backend.zeros((1, 3), dtype=backend.float64)
raw_result = raw_jax.std(
    values,
    axis=0,
    dtype=backend.float64,
    out=raw_out,
    ddof=1,
    keepdims=True,
)
assert tuple(raw_result.shape) == (1, 3)
assert backend.allclose(raw_result, expected)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
