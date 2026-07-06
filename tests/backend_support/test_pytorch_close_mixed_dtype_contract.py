import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


def _skip_without_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")


@pytest.mark.backend_portable
def test_public_pytorch_close_accepts_mixed_integer_tensor_dtypes():
    _skip_without_torch()

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import torch

import pyrecest.backend as backend

left = torch.tensor([1, 0, 2], dtype=torch.uint8)
right = torch.tensor([1, 1, 2], dtype=torch.int16)
expected = np.isclose(
    left.cpu().numpy(),
    right.cpu().numpy(),
    rtol=backend.rtol,
    atol=backend.atol,
)

result = backend.isclose(left, right)
np.testing.assert_array_equal(backend.to_numpy(result), expected)
assert backend.allclose(left, left.to(dtype=torch.int16))
assert not backend.allclose(left, right)
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_runtime_close_patch_accepts_mixed_integer_tensor_dtypes():
    _skip_without_torch()

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import torch

import pyrecest  # noqa: F401
from pyrecest._backend_runtime_patches import patch_pytorch_close_equal_nan_device_contract
import pyrecest._backend.pytorch as raw_pytorch

patch_pytorch_close_equal_nan_device_contract()

left = torch.tensor([1, 0, 2], dtype=torch.uint8)
right = torch.tensor([1, 1, 2], dtype=torch.int16)
expected = np.isclose(
    left.cpu().numpy(),
    right.cpu().numpy(),
    rtol=raw_pytorch.rtol,
    atol=raw_pytorch.atol,
)

result = raw_pytorch.isclose(left, right)
np.testing.assert_array_equal(raw_pytorch.to_numpy(result), expected)
assert raw_pytorch.allclose(left, left.to(dtype=torch.int16))
assert not raw_pytorch.allclose(left, right)
""",
    )

    assert result.returncode == 0, result.stderr
