import math
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, array, cos, eye, linalg, log, ones, pi, sin, to_numpy
from pyrecest.distributions import SO3UniformDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)

ATOL = 1e-6


def scalar(value):
    return float(to_numpy(value).reshape(-1)[0])


def z_quaternion(angle):
    return array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])


class SO3UniformDistributionTest(unittest.TestCase):
    def test_inherits_hyperhemispherical_uniform_distribution(self):
        dist = SO3UniformDistribution()

        self.assertIsInstance(dist, HyperhemisphericalUniformDistribution)
        self.assertEqual(dist.dim, 3)
        self.assertEqual(dist.input_dim, 4)
        self.assertTrue(dist.is_valid())

    def test_pdf_and_ln_pdf_are_constant(self):
        dist = SO3UniformDistribution()
        rotations = array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, sin(pi / 4.0), cos(pi / 4.0)],
            ]
        )
        expected_pdf = ones(3) / (pi**2)

        npt.assert_allclose(dist.pdf(rotations), expected_pdf, atol=ATOL)
        npt.assert_allclose(dist.ln_pdf(rotations), log(expected_pdf), atol=ATOL)
        self.assertAlmostEqual(dist.get_manifold_size(), math.pi**2, places=12)

    def test_pdf_preserves_nested_batch_shape(self):
        dist = SO3UniformDistribution()
        rotations = array(
            [
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, -2.0],
                    [0.0, 0.0, sin(pi / 4.0), cos(pi / 4.0)],
                ],
                [
                    [0.0, 0.0, sin(pi / 6.0), cos(pi / 6.0)],
                    [0.0, sin(pi / 8.0), 0.0, cos(pi / 8.0)],
                    [sin(pi / 10.0), 0.0, 0.0, cos(pi / 10.0)],
                ],
            ]
        )
        expected_pdf = ones((2, 3)) / (pi**2)

        actual_pdf = dist.pdf(rotations)
        actual_ln_pdf = dist.ln_pdf(rotations)

        self.assertEqual(actual_pdf.shape, (2, 3))
        self.assertEqual(actual_ln_pdf.shape, (2, 3))
        npt.assert_allclose(actual_pdf, expected_pdf, atol=ATOL)
        npt.assert_allclose(actual_ln_pdf, log(expected_pdf), atol=ATOL)

    def test_pdf_rejects_nonfinite_quaternions(self):
        dist = SO3UniformDistribution()
        invalid_rotations = [
            array([math.inf, 0.0, 0.0, 1.0]),
            array([math.nan, 0.0, 0.0, 1.0]),
            array([[0.0, 0.0, 0.0, 1.0], [0.0, math.inf, 0.0, 1.0]]),
        ]

        for rotation in invalid_rotations:
            with self.subTest(rotation=rotation):
                with self.assertRaisesRegex(ValueError, "finite"):
                    dist.pdf(rotation)

    def test_sample_returns_canonical_unit_quaternions(self):
        dist = SO3UniformDistribution()

        samples = dist.sample(16)

        self.assertEqual(samples.shape, (16, 4))
        npt.assert_allclose(linalg.norm(samples, None, -1), ones(16), atol=ATOL)
        self.assertTrue(bool(to_numpy(all(samples[:, -1] >= 0.0))))

    def test_geodesic_distance_respects_antipodal_equivalence(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        identity_antipodal = array([0.0, 0.0, 0.0, -1.0])
        quarter_turn = z_quaternion(pi / 2.0)

        npt.assert_allclose(
            SO3UniformDistribution.geodesic_distance(identity, identity_antipodal),
            array([0.0]),
            atol=ATOL,
        )
        npt.assert_allclose(
            SO3UniformDistribution.geodesic_distance(identity, quarter_turn),
            array([pi / 2.0]),
            atol=ATOL,
        )

    def test_rotation_matrix_conversion(self):
        quarter_turn = z_quaternion(pi / 2.0)
        expected = array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        rotation_matrix = SO3UniformDistribution.as_rotation_matrices(quarter_turn)[0]

        npt.assert_allclose(rotation_matrix, expected, atol=ATOL)
        npt.assert_allclose(rotation_matrix.T @ rotation_matrix, eye(3), atol=ATOL)

    def test_mean_and_mode_are_unavailable(self):
        dist = SO3UniformDistribution()

        with self.assertRaises(AttributeError):
            dist.mean()
        with self.assertRaises(AttributeError):
            dist.mode()
        with self.assertRaises(AttributeError):
            dist.mean_axis()


if __name__ == "__main__":
    unittest.main()
