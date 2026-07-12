import unittest

import matplotlib
import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, column_stack, cos, exp, pi, sin
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import (
    ToroidalVonMisesSineDistribution,
)
from scipy.special import ive

matplotlib.pyplot.close("all")
matplotlib.use("Agg")


class ToroidalBivarVMTestMixin:
    """Shared tests for bivariate toroidal von Mises distribution test classes.

    Subclasses must implement ``setUp`` (setting ``self.mu``, ``self.kappa``,
    and ``self.tvm``) and ``_unnormalized_pdf``.
    """

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.tvm.integrate(), 1.0, delta=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_trigonometric_moment_numerical(self):
        npt.assert_allclose(
            self.tvm.trigonometric_moment_numerical(0), array([1.0, 1.0])
        )

    def test_plot_2d(self):
        self.tvm.plot()

    @parameterized.expand(
        [
            (array([3.0, 2.0]),),
            (array([1.0, 4.0]),),
            (array([5.0, 6.0]),),
            (array([-3.0, 11.0]),),
            (array([[5.0, 1.0], [6.0, 3.0]]),),
            (
                column_stack(
                    (arange(0.0, 2.0 * pi, 0.1), arange(1.0 * pi, 3.0 * pi, 0.1))
                ),
            ),
        ]
    )
    def test_pdf(self, x):
        C = self.tvm.C

        def pdf(x):
            return self._unnormalized_pdf(x) * C

        expected = pdf(x)

        npt.assert_allclose(self.tvm.pdf(x), expected)

    def test_pdf_accepts_list_inputs(self):
        x = [[1.0, 2.0], [1.3, 2.4]]

        npt.assert_allclose(self.tvm.pdf(x), self.tvm.pdf(array(x)))
        npt.assert_allclose(self.tvm.pdf([1.3, 2.4]), self.tvm.pdf(array([1.3, 2.4])))

    def test_pdf_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            self.tvm.pdf([1.0, 2.0, 3.0])


class ToroidalVMSineDistributionTest(ToroidalBivarVMTestMixin, unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.kappa = array([0.7, 1.4])
        self.lambda_ = array(0.5)
        self.tvm = ToroidalVonMisesSineDistribution(self.mu, self.kappa, self.lambda_)

    def test_instance(self):
        # sanity check
        self.assertIsInstance(self.tvm, ToroidalVonMisesSineDistribution)

    def test_mu_kappa_lambda(self):
        npt.assert_allclose(self.tvm.mu, self.mu)
        npt.assert_allclose(self.tvm.kappa, self.kappa)
        self.assertEqual(self.tvm.lambda_, self.lambda_)

    def test_accepts_list_parameters(self):
        tvm = ToroidalVonMisesSineDistribution([1.0, 2.0], [0.7, 1.4], 0.5)

        npt.assert_allclose(tvm.mu, self.mu)
        npt.assert_allclose(tvm.kappa, self.kappa)
        npt.assert_allclose(tvm.lambda_, self.lambda_)

    def test_accepts_python_scalar_lambda(self):
        tvm = ToroidalVonMisesSineDistribution(self.mu, self.kappa, 0.5)
        x = array([1.3, 2.4])

        npt.assert_allclose(tvm.pdf(x), self.tvm.pdf(x), rtol=5e-7)
        npt.assert_allclose(tvm.norm_const, self.tvm.norm_const, rtol=5e-7)

    def test_constructor_rejects_invalid_parameters(self):
        invalid_cases = [
            ([1.0], self.kappa, self.lambda_, "mu"),
            (self.mu, [0.7], self.lambda_, "kappa"),
            (self.mu, [-0.7, 1.4], self.lambda_, "nonnegative"),
            (self.mu, [float("nan"), 1.4], self.lambda_, "finite"),
            (self.mu, self.kappa, [0.5, 0.6], "lambda_"),
            (self.mu, self.kappa, float("nan"), "lambda_"),
        ]

        for mu, kappa, lambda_, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ToroidalVonMisesSineDistribution(mu, kappa, lambda_)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",
        reason="Regression test uses NumPy/scipy scalar semantics",
    )
    def test_zero_kappa_normalizer_is_finite(self):
        tvm = ToroidalVonMisesSineDistribution(
            array([1.0, 2.0]), array([0.0, 0.0]), array(0.5)
        )
        self.assertTrue(np.isfinite(tvm.norm_const))
        self.assertAlmostEqual(tvm.integrate(), 1.0, delta=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",
        reason="Regression test uses NumPy/scipy scalar semantics",
    )
    def test_large_kappa_independent_mode_density_is_finite(self):
        kappa = 1000.0
        mu = array([0.4, 1.2])
        tvm = ToroidalVonMisesSineDistribution(mu, array([kappa, kappa]), 0.0)

        mode_density = tvm.pdf(mu)
        expected = 1.0 / (4.0 * np.pi**2 * ive(0, kappa) ** 2)

        self.assertTrue(np.isfinite(mode_density))
        self.assertGreater(float(mode_density), 0.0)
        npt.assert_allclose(mode_density, expected, rtol=1e-12)

    def _unnormalized_pdf(self, xs):
        return exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self.lambda_ * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
        )


if __name__ == "__main__":
    unittest.main()
