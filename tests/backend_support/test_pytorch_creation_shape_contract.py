import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_public_pytorch_creation_helpers_reject_boolean_shapes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

for call in (
    lambda: backend.empty(True),
    lambda: backend.zeros([True, False]),
    lambda: backend.ones(backend.asarray(True)),
    lambda: backend.full([False], 1.0),
):
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError("boolean shape was accepted")

assert tuple(backend.zeros([]).shape) == ()
assert tuple(backend.ones(backend.asarray(2)).shape) == (2,)
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_creation_helpers_reject_boolean_shapes_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import pyrecest
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

assert backend.__backend_name__ == "numpy"

for call in (
    lambda: pytorch_backend.empty(True),
    lambda: pytorch_backend.zeros([True, False]),
    lambda: pytorch_backend.ones(pytorch_backend.asarray(True)),
    lambda: pytorch_backend.full([False], 1.0),
):
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError("boolean shape was accepted")

assert tuple(pytorch_backend.zeros([]).shape) == ()
assert tuple(pytorch_backend.ones(pytorch_backend.asarray(2)).shape) == (2,)
""",
    )

    assert result.returncode == 0, result.stderr
