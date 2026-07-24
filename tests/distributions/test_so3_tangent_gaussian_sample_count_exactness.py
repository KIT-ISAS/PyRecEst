import unittest
from fractions import Fraction

from pyrecest.distributions.so3_tangent_gaussian_distribution import (
    _validate_positive_sample_count,
)


class SO3TangentGaussianSampleCountExactnessTest(unittest.TestCase):
    def test_rejects_fractional_count_rounded_to_integer_by_binary64(self):
        rounded_half_integer = Fraction(2**54 + 1, 2)

        with self.assertRaises(ValueError):
            _validate_positive_sample_count(rounded_half_integer)

    def test_preserves_exact_large_integral_count(self):
        exact_integer = Fraction(2**54 + 2, 2)

        self.assertEqual(
            _validate_positive_sample_count(exact_integer),
            2**53 + 1,
        )


if __name__ == "__main__":
    unittest.main()
