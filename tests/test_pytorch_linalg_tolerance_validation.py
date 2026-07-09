import unittest
from typing import Any

import numpy as np

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchLinalgToleranceValidation(unittest.TestCase):
    def _test_matrix(self):
        return pytorch_backend.diag(
            pytorch_backend.array([1.0, 1e-5], dtype=pytorch_backend.float64)
        )

    def _invalid_tolerances(self):
        return [
            True,
            np.bool_(True),
            np.array(True),
            np.array([True]),
            np.array([True], dtype=object),
            pytorch_backend.array(True),
            np.timedelta64(1, "ns"),
            np.array(np.timedelta64(1, "ns")),
            np.array([np.timedelta64(1, "ns")]),
            np.array([np.timedelta64(1, "ns")], dtype=object),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.array(np.datetime64("1970-01-01T00:00:00.000000001")),
            np.array([np.datetime64("1970-01-01T00:00:00.000000001")]),
            np.array(
                [np.datetime64("1970-01-01T00:00:00.000000001")],
                dtype=object,
            ),
        ]

    def test_matrix_rank_rejects_boolean_and_temporal_tolerances(self):
        value = self._test_matrix()

        for keyword in ("tol", "atol", "rtol"):
            for tolerance in self._invalid_tolerances():
                with self.subTest(keyword=keyword, tolerance=repr(tolerance)):
                    with self.assertRaises(TypeError):
                        pytorch_backend.linalg.matrix_rank(
                            value,
                            **{keyword: tolerance},
                        )

    def test_pinv_rejects_boolean_and_temporal_tolerances(self):
        value = self._test_matrix()

        for keyword in ("rcond", "atol", "rtol"):
            for tolerance in self._invalid_tolerances():
                with self.subTest(keyword=keyword, tolerance=repr(tolerance)):
                    with self.assertRaises(TypeError):
                        pytorch_backend.linalg.pinv(
                            value,
                            **{keyword: tolerance},
                        )


if __name__ == "__main__":
    unittest.main()
