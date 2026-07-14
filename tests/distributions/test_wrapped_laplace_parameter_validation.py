import unittest

from pyrecest.backend import array
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


class WrappedLaplaceParameterValidationTest(unittest.TestCase):
    def test_rejects_non_real_positive_scalar_parameters(self):
        invalid_values = (True, array([True]), "2.0", 1.0 + 0.0j)
        for invalid in invalid_values:
            with self.subTest(lambda_=invalid), self.assertRaisesRegex(
                ValueError, "lambda_"
            ):
                WrappedLaplaceDistribution(invalid, 1.0)
            with self.subTest(kappa=invalid), self.assertRaisesRegex(
                ValueError, "kappa_"
            ):
                WrappedLaplaceDistribution(1.0, invalid)


if __name__ == "__main__":
    unittest.main()
