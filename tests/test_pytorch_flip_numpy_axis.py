from tests.support.backend_runner import run_backend_code


def test_pytorch_flip_accepts_numpy_integer_axis():
    code = '''
import numpy as np
import pyrecest.backend as backend

result = backend.flip([1, 2, 3], axis=np.int64(0))
assert backend.to_numpy(result).tolist() == [3, 2, 1]
'''
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
