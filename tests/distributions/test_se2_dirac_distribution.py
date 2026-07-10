import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,duplicate-code
import pyrecest.backend

# pylint: disable=redefined-builtin
from pyrecest.backend import array, ones, pi, random, sum
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)
from pyrecest.distributions.se2_dirac_distribution import SE2DiracDistribution


class TestSE2DiracDistribution(unittest.TestCase):
    def setUp(self):
        self.d = array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2.0, 4.0, 0.0, 0.5, 1.0, 1.0],
                [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            ]
        ).T
        self.w = array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        self.w = self.w / sum(self.w)
        self.dist = SE2DiracDistribution(self.d, self.w)

    def test_init_uniform_weights(self):
        dist = SE2DiracDistribution(self.d)
        n = self.d.shape[0]
        npt.assert_allclose(dist.w, ones(n) / n)

    def test_bound_and_lin_dim(self):
        self.assertEqual(self.dist.bound_dim, 1)
        self.assertEqual(self.dist.lin_dim, 2)

    def test_mean_4d_matches_hybrid_moment(self):
        npt.assert_array_equal(self.dist.mean_4d(), self.dist.hybrid_moment())

    def test_mean_4d_values(self):
        m = self.dist.mean_4d()
        self.assertEqual(m.shape, (4,))
        # First two components are weighted cos/sin of angles
        from pyrecest.backend import cos, sin  # pylint: disable=import-outside-toplevel

        expected_cos = sum(self.w * cos(self.d[:, 0]))
        expected_sin = sum(self.w * sin(self.d[:, 0]))
        npt.assert_allclose(m[0], expected_cos, rtol=1e-10)
        npt.assert_allclose(m[1], expected_sin, rtol=1e-10)
        # Last two are weighted linear means
        npt.assert_allclose(m[2], sum(self.w * self.d[:, 1]), rtol=1e-10)
        npt.assert_allclose(m[3], sum(self.w * self.d[:, 2]), rtol=1e-10)

    def test_covariance_4d_shape(self):
        C = self.dist.covariance_4d()
        self.assertEqual(C.shape, (4, 4))

    def test_covariance_4d_symmetric(self):
        C = self.dist.covariance_4d()
        npt.assert_allclose(C, C.T, atol=1e-7)

    def test_covariance_4d_positive_semidefinite(self):
        C = np.array(self.dist.covariance_4d())
        eigvals = np.linalg.eigvalsh(C)
        self.assertTrue(np.all(eigvals >= -1e-12))

    def test_covariance_4d_matches_weighted_centered_moment(self):
        samples = np.column_stack(
            (
                np.cos(np.asarray(self.d[:, 0])),
                np.sin(np.asarray(self.d[:, 0])),
                np.asarray(self.d[:, 1:]),
            )
        )
        weights = np.asarray(self.w)
        mean = np.sum(weights[:, None] * samples, axis=0)
        centered = samples - mean
        expected = centered.T @ (weights[:, None] * centered)

        npt.assert_allclose(self.dist.covariance_4d(), expected, rtol=1e-10)

    def test_covariance_4d_is_zero_for_single_component(self):
        dist = SE2DiracDistribution(array([[0.5, 2.0, -3.0]]), array([1.0]))

        npt.assert_allclose(dist.covariance_4d(), np.zeros((4, 4)), atol=1e-12)

    def test_mean_delegates_to_hybrid_mean(self):
        npt.assert_array_equal(self.dist.mean(), self.dist.hybrid_mean())

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_from_distribution(self):
        random.seed(0)
        mu = array([1.0, 2.0, 3.0])
        C = array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        pwn = PartiallyWrappedNormalDistribution(mu, C, bound_dim=1)
        ddist = SE2DiracDistribution.from_distribution(pwn, 50000)
        self.assertIsInstance(ddist, SE2DiracDistribution)
        npt.assert_allclose(ddist.hybrid_mean(), pwn.hybrid_mean(), atol=0.05)

    def test_from_distribution_type_error(self):
        with self.assertRaises(TypeError):
            SE2DiracDistribution.from_distribution("not_a_distribution", 10)

    def test_from_distribution_particles_error(self):
        mu = array([1.0, 2.0, 3.0])
        C = array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        pwn = PartiallyWrappedNormalDistribution(mu, C, bound_dim=1)

        ddist = SE2DiracDistribution.from_distribution(pwn, np.int64(3))

        self.assertEqual(ddist.d.shape, (3, 3))
        self.assertEqual(ddist.w.shape, (3,))

        for n_particles in (True, 1.5, 0, -1):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    SE2DiracDistribution.from_distribution(pwn, n_particles)

    def test_marginalize_linear(self):
        from pyrecest.distributions.hypertorus.hypertoroidal_dirac_distribution import (  # pylint: disable=import-outside-toplevel
            HypertoroidalDiracDistribution,
        )

        wd = self.dist.marginalize_linear()
        self.assertIsInstance(wd, HypertoroidalDiracDistribution)
        # The trigonometric moment of the marginalized distribution matches mean_4d
        m = self.dist.mean_4d()
        npt.assert_allclose(m[0], wd.trigonometric_moment(1).real, rtol=1e-10)
        npt.assert_allclose(m[1], wd.trigonometric_moment(1).imag, rtol=1e-10)

    def test_sampling_accepts_scalar_array_count(self):
        random.seed(0)
        samples = self.dist.sample(np.array(3, dtype=np.int64))

        self.assertEqual(samples.shape, (3, 3))

    def test_sampling_rejects_invalid_count_values(self):
        for n in (np.array([3], dtype=np.int64), np.array(True), True, 1.5, 0, -1):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.dist.sample(n)

    def test_sampling(self):
        random.seed(0)
        n = 20
        s = self.dist.sample(n)
        self.assertEqual(s.shape, (n, 3))
        # Angles should be in [0, 2*pi)
        from pyrecest.backend import (
            all as backend_all,  # pylint: disable=import-outside-toplevel
        )

        self.assertTrue(backend_all(s[:, 0] >= 0))
        self.assertTrue(backend_all(s[:, 0] < 2 * pi))


if __name__ == "__main__":
    unittest.main()
