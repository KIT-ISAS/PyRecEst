import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_flip_accepts_numpy_integer_axis():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = '''
import numpy as np
import pyrecest.backend as backend

result = backend.flip([1, 2, 3], axis=np.int64(0))
assert backend.to_numpy(result).tolist() == [3, 2, 1]
'''
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
