import unittest

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution


class GaussianMarginalizeOutValidationTest(unittest.TestCase):
    def setUp(self):
        self.distribution = GaussianDistribution(
            array([1.0, 2.0]),
            array([[1.1, 0.4], [0.4, 0.9]]),
        )

    def test_rejects_duplicate_dimensions(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.distribution.marginalize_out([0, 0])

    def test_rejects_empty_marginal(self):
        with self.assertRaisesRegex(ValueError, "leave at least one dimension"):
            self.distribution.marginalize_out([0, 1])


if __name__ == "__main__":
    unittest.main()
