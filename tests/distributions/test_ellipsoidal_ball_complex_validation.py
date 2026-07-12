import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution
from pyrecest.exceptions import ValidationError


class TestEllipsoidalBallComplexValidation(unittest.TestCase):
    def test_rejects_complex_center(self):
        with self.assertRaisesRegex(ValidationError, "center must be real-valued"):
            EllipsoidalBallUniformDistribution(
                array([0.0 + 0.0j, 0.0 + 0.0j]),
                diag(array([1.0, 1.0])),
            )

    def test_rejects_complex_shape_matrix(self):
        with self.assertRaisesRegex(
            ValidationError, "shape_matrix must be real-valued"
        ):
            EllipsoidalBallUniformDistribution(
                array([0.0, 0.0]),
                array([[1.0 + 0.0j, 0.5j], [0.5j, 1.0 + 0.0j]]),
            )

    def test_pdf_rejects_complex_points(self):
        distribution = EllipsoidalBallUniformDistribution(
            array([0.0]),
            diag(array([1.0])),
        )

        with self.assertRaisesRegex(ValidationError, "xs must be real-valued"):
            distribution.pdf(array([1.0j]))


if __name__ == "__main__":
    unittest.main()
