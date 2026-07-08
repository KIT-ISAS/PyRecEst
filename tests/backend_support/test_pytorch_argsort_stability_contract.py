"""Regression tests for PyTorch argsort keyword compatibility."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_subprocess_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_public_pytorch_argsort_rejects_kind_and_stable_combinations():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("pytorch")
    code = """
import numpy as np
import pyrecest.backend as backend

assert backend.__backend_name__ == "pytorch"
values = backend.asarray([2, 1, 1])
stable_values = (True, False, np.bool_(True), np.bool_(False), 1, 0)

for kind in ("stable", "mergesort", "quicksort", "heapsort"):
    for stable in stable_values:
        try:
            backend.argsort(values, kind=kind, stable=stable)
        except ValueError as exc:
            assert "conflicting" in str(exc)
        else:
            raise AssertionError(f"accepted kind={kind!r}, stable={stable!r}")

assert backend.to_numpy(backend.argsort(values, kind="stable")).tolist() == [1, 2, 0]
assert backend.to_numpy(backend.argsort(values, stable=True)).tolist() == [1, 2, 0]
assert backend.to_numpy(backend.argsort([[3, 2], [1, 0]], dim=1)).tolist() == [
    [1, 0],
    [1, 0],
]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
