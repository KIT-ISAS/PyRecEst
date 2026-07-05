import importlib.util
import os
import subprocess
import sys

import pytest


def _run_backend_code(backend_name, code):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_pytorch_argsort_accepts_numpy_axis_contract():
    _run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

axis_result = backend.argsort([[3, 1, 2], [0, 5, 4]], axis=1)
assert backend.to_numpy(axis_result).tolist() == [[1, 2, 0], [0, 2, 1]]

flat_result = backend.argsort([[3, 1], [0, 2]], axis=None)
assert backend.to_numpy(flat_result).tolist() == [2, 1, 3, 0]

dim_result = backend.argsort([[3, 1], [0, 2]], dim=0)
assert backend.to_numpy(dim_result).tolist() == [[1, 0], [0, 1]]

stable_result = backend.argsort([2, 1, 2], stable=True)
assert backend.to_numpy(stable_result).tolist() == [1, 0, 2]

raw_result = pytorch_backend.argsort([[2, 0], [1, 3]], axis=0)
assert pytorch_backend.to_numpy(raw_result).tolist() == [[1, 0], [0, 1]]

try:
    backend.argsort([1, 2], axis=0, dim=1)
except TypeError:
    pass
else:
    raise AssertionError("argsort accepted conflicting axis and dim arguments")
""",
    )


@pytest.mark.backend_portable
def test_raw_pytorch_argsort_is_patched_under_numpy_backend():
    _run_backend_code(
        "numpy",
        """
import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

assert backend.__backend_name__ == "numpy"

raw_axis_result = pytorch_backend.argsort([[3, 1, 2], [0, 5, 4]], axis=1)
assert pytorch_backend.to_numpy(raw_axis_result).tolist() == [[1, 2, 0], [0, 2, 1]]

raw_flat_result = pytorch_backend.argsort([[3, 1], [0, 2]], axis=None)
assert pytorch_backend.to_numpy(raw_flat_result).tolist() == [2, 1, 3, 0]

public_result = backend.argsort([[3, 1, 2], [0, 5, 4]], axis=1)
assert public_result.tolist() == [[1, 2, 0], [0, 2, 1]]

try:
    pytorch_backend.argsort([1, 2], axis=0, dim=1)
except TypeError:
    pass
else:
    raise AssertionError("raw argsort accepted conflicting axis and dim arguments")
""",
    )
