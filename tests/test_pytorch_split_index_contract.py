import unittest
from fractions import Fraction

from pyrecest.backend_support._pytorch_split_index_contract import (
    _normalize_split_section_count,
)


class _TorchStub:
    @staticmethod
    def is_tensor(value):
        del value
        return False


class PytorchSplitIndexContractTest(unittest.TestCase):
    def test_rejects_fractional_count_hidden_by_float_rounding(self):
        fractional_count = Fraction(2**54 + 1, 2)

        with self.assertRaisesRegex(ValueError, "number sections must be an integer"):
            _normalize_split_section_count(fractional_count, _TorchStub)

    def test_preserves_large_exact_integral_count(self):
        exact_count = Fraction(2**54 + 2, 2)

        normalized = _normalize_split_section_count(exact_count, _TorchStub)

        self.assertEqual(normalized, 2**53 + 1)


if __name__ == "__main__":
    unittest.main()
