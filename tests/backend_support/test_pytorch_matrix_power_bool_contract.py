import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_matrix_power_rejects_boolean_exponents():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import torch
from pyrecest.backend import linalg
import pyrecest._backend.pytorch.linalg as raw_linalg

matrix = [[1.0, 1.0], [0.0, 1.0]]
boolean_exponents = [
    True,
    False,
    np.bool_(True),
    np.array(True),
    torch.tensor(True),
]

for exponent in boolean_exponents:
    for linalg_module in (linalg, raw_linalg):
        try:
            linalg_module.matrix_power(matrix, exponent)
        except TypeError as exc:
            assert "boolean" in str(exc)
        else:
            raise AssertionError(
                f"matrix_power accepted boolean exponent {exponent!r} via {linalg_module!r}"
            )

valid_result = raw_linalg.matrix_power(matrix, np.array(2))
assert valid_result.detach().cpu().numpy().tolist() == [[1.0, 2.0], [0.0, 1.0]]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
