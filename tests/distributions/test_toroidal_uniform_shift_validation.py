import unittest

from pyrecest.distributions.hypertorus.toroidal_uniform_distribution import (
    ToroidalUniformDistribution,
)


class TestToroidalUniformShiftValidation(unittest.TestCase):
    def test_shift_rejects_wrong_dimension(self):
        dist = ToroidalUniformDistribution()

        for shift_by in (1.0, [1.0], [1.0, 2.0, 3.0], [[1.0, 2.0]]):
            with self.subTest(shift_by=shift_by):
                with self.assertRaisesRegex(ValueError, "shift_by"):
                    dist.shift(shift_by)


if __name__ == "__main__":
    unittest.main()
