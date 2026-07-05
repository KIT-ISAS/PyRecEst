import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import abs, allclose, array, linspace
from pyrecest.distributions import VonMisesDistribution


class TestVonMisesDistribution(unittest.TestCase):
    def test_vm_init(self):
        dist1 = VonMisesDistribution(0, 1)
        dist2 = VonMisesDistribution(2, 1)
        self.assertEqual(dist1.kappa, dist2.kappa)
        self.assertNotEqual(dist1.mu, dist2.mu)

    def test_constructor_rejects_invalid_kappa(self):
        for kappa in (-1.0, float("nan"), float("inf"), [1.0, 2.0]):
            with self.subTest(kappa=kappa):
                with self.assertRaises(ValueError):
                    VonMisesDistribution(0.0, kappa)

    def test_set_mean_returns_shifted_copy(self):
        dist = VonMisesDistribution(array(0.3), array(2.0))

        shifted = dist.set_mean(array(1.1))

        self.assertIsInstance(shifted, VonMisesDistribution)
        self.assertIsNot(shifted, dist)
        npt.assert_allclose(float(dist.mu), 0.3)
        npt.assert_allclose(float(shifted.mu), 1.1)
        npt.assert_allclose(float(shifted.kappa), 2.0)

    def test_besselratio_remains_finite_for_large_concentration(self):
        ratio = VonMisesDistribution.besselratio(0, 1000.0)

        self.assertTrue(np.isfinite(float(ratio)))
        self.assertGreater(float(ratio), 0.999)
        self.assertLess(float(ratio), 1.0)

    def test_large_concentration_pdf_and_entropy_are_finite(self):
        dist = VonMisesDistribution(0.3, 1000.0)

        mode_pdf = dist.pdf(array([0.3]))
        entropy = dist.entropy()

        self.assertTrue(np.isfinite(float(mode_pdf[0])))
        self.assertGreater(float(mode_pdf[0]), 0.0)
        self.assertTrue(np.isfinite(float(entropy)))

    def test_large_concentration_pdf_stays_stable_after_norm_const_access(self):
        dist = VonMisesDistribution(0.3, 1000.0)
        expected = VonMisesDistribution(0.3, 1000.0).pdf(array([0.3]))

        _ = dist.norm_const
        mode_pdf = dist.pdf(array([0.3]))

        self.assertTrue(np.isfinite(float(mode_pdf[0])))
        npt.assert_allclose(mode_pdf, expected)

    def test_pdf(self):
        dist = VonMisesDistribution(2, 1)
        xs = linspace(1, 7, 7)
        npt.assert_array_almost_equal(
            dist.pdf(xs),
            array(
                [
                    0.215781465110296,
                    0.341710488623463,
                    0.215781465110296,
                    0.0829150854731715,
                    0.0467106111086458,
                    0.0653867888824553,
                    0.166938593220285,
                ],
            ),
        )

    def test_pdf_accepts_list_inputs(self):
        dist = VonMisesDistribution(0.3, 1.2)
        xs = [0.5, 1.0]

        npt.assert_allclose(dist.pdf(xs), dist.pdf(array(xs)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf_accepts_scalar_and_list_inputs(self):
        dist = VonMisesDistribution(0.3, 1.2)
        xs = [0.5, 1.0]

        npt.assert_allclose(dist.cdf(0.5), dist.cdf(array(0.5)))
        npt.assert_allclose(dist.cdf(xs), dist.cdf(array(xs)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf_rejects_matrix_inputs(self):
        dist = VonMisesDistribution(0.3, 1.2)

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            dist.cdf(array([[0.5, 1.0]]))

    def test_uniform_trigonometric_moments(self):
        dist = VonMisesDistribution(2, 0)

        npt.assert_allclose(dist.trigonometric_moment(0), array(1.0 + 0.0j))
        npt.assert_allclose(dist.trigonometric_moment(1), array(0.0 + 0.0j))
        npt.assert_allclose(dist.trigonometric_moment(2), array(0.0 + 0.0j))

    def test_from_moment_recovers_parameters(self):
        dist = VonMisesDistribution(1.3, 4.0)
        reconstructed = VonMisesDistribution.from_moment(dist.trigonometric_moment(1))

        self.assertAlmostEqual(float(reconstructed.mu), float(dist.mu))
        self.assertAlmostEqual(float(reconstructed.kappa), float(dist.kappa))

    def test_from_zero_moment_returns_uniform_distribution(self):
        dist = VonMisesDistribution.from_moment(array(0.0 + 0.0j))

        self.assertEqual(dist.mu, 0.0)
        self.assertEqual(dist.kappa, 0.0)

    def test_from_degenerate_moment_raises(self):
        with self.assertRaises(ValueError):
            VonMisesDistribution.from_moment(array(1.0 + 0.0j))

    def test_from_invalid_moment_raises(self):
        with self.assertRaises(ValueError):
            VonMisesDistribution.from_moment(array(1.01 + 0.0j))

    def test_besselratio_inverse_rejects_invalid_boundary_values(self):
        for x in (-0.1, 1.0, 1.01):
            with self.assertRaises(ValueError):
                VonMisesDistribution.besselratio_inverse(0, x)

    def test_convolution_of_uniform_components_remains_uniform(self):
        first = VonMisesDistribution(array(0.3), array(0.0))
        second = VonMisesDistribution(array(0.4), array(2.0))

        convolved = first.convolve(second)

        self.assertTrue(allclose(convolved.kappa, array(0.0), atol=1e-14))
        self.assertTrue(allclose(abs(convolved.trigonometric_moment(1)), array(0.0)))

    def test_sample_accepts_integer_like_count(self):
        dist = VonMisesDistribution(array(0.3), array(1.0))

        samples = dist.sample(np.int64(4))

        self.assertEqual(samples.shape, (4,))

    def test_sample_rejects_invalid_count(self):
        dist = VonMisesDistribution(array(0.3), array(1.0))

        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    dist.sample(n)

    def test_plot(self):
        matplotlib.pyplot.close("all")
        matplotlib.use("Agg")
        vm = VonMisesDistribution(0, 1)
        vm.plot()
        plt.close()


if __name__ == "__main__":
    unittest.main()
