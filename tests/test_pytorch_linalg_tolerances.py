import unittest
from typing import Any

import numpy as np

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchLinalgToleranceContract(unittest.TestCase):
    @staticmethod
    def _temporal_tolerances():
        return (
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.array(np.timedelta64(1, "ns")),
            np.array(np.datetime64("1970-01-01T00:00:00.000000001")),
        )

    def test_matrix_rank_accepts_numpy_scalar_array_tolerances(self):
        value = pytorch_backend.diag(pytorch_backend.array([1.0, 1e-5]))

        self.assertEqual(
            int(pytorch_backend.linalg.matrix_rank(value, tol=np.array(1e-4))),
            1,
        )
        self.assertEqual(
            int(pytorch_backend.linalg.matrix_rank(value, rtol=np.array(1e-4))),
            1,
        )

    def test_matrix_rank_accepts_numpy_vector_tolerances(self):
        single = pytorch_backend.diag(
            pytorch_backend.array([1.0, 1e-5], dtype=pytorch_backend.float64)
        )
        value = pytorch_backend.stack([single, single])

        result = pytorch_backend.linalg.matrix_rank(
            value,
            rtol=np.array([1e-4, 1e-6]),
        )

        self.assertEqual(pytorch_backend.to_numpy(result).tolist(), [1, 2])

    def test_matrix_rank_rejects_numpy_temporal_tolerances(self):
        value = pytorch_backend.diag(
            pytorch_backend.array([1.0, 1e-5], dtype=pytorch_backend.float64)
        )

        for keyword in ("tol", "rtol", "atol"):
            for tolerance in self._temporal_tolerances():
                with self.subTest(keyword=keyword, tolerance=repr(tolerance)):
                    with self.assertRaises(TypeError):
                        pytorch_backend.linalg.matrix_rank(value, **{keyword: tolerance})

    def test_pinv_accepts_numpy_scalar_array_tolerances(self):
        value = pytorch_backend.diag(
            pytorch_backend.array([1.0, 1e-5], dtype=pytorch_backend.float64)
        )
        expected = pytorch_backend.diag(
            pytorch_backend.array([1.0, 0.0], dtype=pytorch_backend.float64)
        )

        for keyword in ("rcond", "rtol"):
            with self.subTest(keyword=keyword):
                result = pytorch_backend.linalg.pinv(
                    value,
                    **{keyword: np.array(1e-4)},
                )

                self.assertEqual(result.dtype, pytorch_backend.float64)
                self.assertTrue(pytorch_backend.allclose(result, expected))

    def test_pinv_accepts_numpy_vector_tolerances(self):
        single = pytorch_backend.diag(
            pytorch_backend.array([1.0, 1e-5], dtype=pytorch_backend.float64)
        )
        value = pytorch_backend.stack([single, single])
        expected = pytorch_backend.stack(
            [
                pytorch_backend.diag(
                    pytorch_backend.array([1.0, 0.0], dtype=pytorch_backend.float64)
                ),
                pytorch_backend.diag(
                    pytorch_backend.array([1.0, 1e5], dtype=pytorch_backend.float64)
                ),
            ]
        )

        for keyword in ("rcond", "rtol"):
            with self.subTest(keyword=keyword):
                result = pytorch_backend.linalg.pinv(
                    value,
                    **{keyword: np.array([1e-4, 1e-6])},
                )

                self.assertEqual(result.dtype, pytorch_backend.float64)
                self.assertTrue(pytorch_backend.allclose(result, expected))

    def test_pinv_rejects_numpy_temporal_tolerances(self):
        value = pytorch_backend.diag(
            pytorch_backend.array([1.0, 1e-5], dtype=pytorch_backend.float64)
        )

        for keyword in ("rcond", "rtol", "atol"):
            for tolerance in self._temporal_tolerances():
                with self.subTest(keyword=keyword, tolerance=repr(tolerance)):
                    with self.assertRaises(TypeError):
                        pytorch_backend.linalg.pinv(value, **{keyword: tolerance})


if __name__ == "__main__":
    unittest.main()
