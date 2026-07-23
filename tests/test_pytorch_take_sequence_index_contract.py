import unittest

import numpy as np

from pyrecest.backend_support._pytorch_take_index_contract import (
    _validate_take_indices,
)


class _TorchStub:
    @staticmethod
    def is_tensor(value):
        del value
        return False


class PytorchTakeSequenceIndexContractTest(unittest.TestCase):
    def test_rejects_invalid_plain_python_index_sequences(self):
        invalid_indices = (
            [0.0, 1.0],
            (0.0, 1.0),
            [0.0 + 0.0j, 1.0 + 0.0j],
            ["0", "1"],
        )

        for indices in invalid_indices:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(
                    TypeError,
                    "indices must be integers or boolean values",
                ):
                    _validate_take_indices(indices, _TorchStub)

    def test_preserves_valid_plain_python_index_sequences(self):
        valid_indices = ([0, 2], (1, 0), [True, False])

        for indices in valid_indices:
            with self.subTest(indices=indices):
                self.assertIs(_validate_take_indices(indices, _TorchStub), indices)

    def test_preserves_scalar_indices_for_existing_scalar_handling(self):
        scalar = np.float64(1.0)

        self.assertIs(_validate_take_indices(scalar, _TorchStub), scalar)


if __name__ == "__main__":
    unittest.main()
