import unittest

import numpy as np

jax_backend = None
try:
    from pyrecest._backend import jax as jax_backend
except ModuleNotFoundError:
    pass


@unittest.skipIf(jax_backend is None, "JAX is not installed")
class TestJaxFftBoolContract(unittest.TestCase):
    def test_real_fft_rejects_python_bool_axis(self):
        with self.assertRaisesRegex(TypeError, "axis must be an integer"):
            jax_backend.fft.rfft([1.0, 2.0], axis=True)

    def test_real_fft_rejects_numpy_bool_axis(self):
        with self.assertRaisesRegex(TypeError, "axis must be an integer"):
            jax_backend.fft.rfft([1.0, 2.0], axis=np.array(False))

    def test_real_fft_rejects_python_bool_length(self):
        with self.assertRaisesRegex(TypeError, "n must be an integer length"):
            jax_backend.fft.rfft([1.0, 2.0], n=True)

    def test_inverse_real_fft_rejects_numpy_bool_length(self):
        with self.assertRaisesRegex(TypeError, "n must be an integer length"):
            jax_backend.fft.irfft([1.0, 0.0], n=np.array(True))


if __name__ == "__main__":
    unittest.main()
