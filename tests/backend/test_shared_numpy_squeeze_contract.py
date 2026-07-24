import unittest

import numpy as np

import pyrecest.backend as backend


@unittest.skipUnless(
    backend.__backend_name__ in {"autograd", "numpy"},
    reason="Shared NumPy squeeze contract only applies to NumPy-style backends",
)
class SharedNumPySqueezeContractTest(unittest.TestCase):
    def test_explicit_nonsingleton_axis_is_rejected(self):
        values = backend.array([[1.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "size not equal to one"):
            backend.squeeze(values, axis=1)

    def test_singleton_axis_is_squeezed(self):
        values = backend.array([[1.0, 2.0]])

        result = backend.squeeze(values, axis=0)

        np.testing.assert_array_equal(backend.to_numpy(result), np.array([1.0, 2.0]))

    def test_out_of_bounds_axis_remains_rejected(self):
        values = backend.array([[1.0, 2.0]])

        with self.assertRaises((IndexError, ValueError)):
            backend.squeeze(values, axis=2)


if __name__ == "__main__":
    unittest.main()
