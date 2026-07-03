import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_raw_pytorch_reshape_accepts_array_like_inputs_when_public_backend_is_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import numpy.testing as npt
import pyrecest  # noqa: F401
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"

result = raw_pytorch.reshape([1, 2, 3, 4], np.array([2, 2]))
assert tuple(result.shape) == (2, 2)
npt.assert_array_equal(raw_pytorch.to_numpy(result), np.array([[1, 2], [3, 4]]))

flat = raw_pytorch.reshape([[1, 2], [3, 4]], np.array(4, dtype=np.int64))
assert tuple(flat.shape) == (4,)
npt.assert_array_equal(raw_pytorch.to_numpy(flat), np.array([1, 2, 3, 4]))

for bad_shape in ("2", [2, 2.0]):
    try:
        raw_pytorch.reshape([1, 2, 3, 4], bad_shape)
    except TypeError:
        pass
    else:
        raise AssertionError(f"reshape accepted invalid shape {bad_shape!r}")

print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
