import unittest

from pyrecest.sampling import MerweScaledSigmaPoints


class TestMerweExtremeScale(unittest.TestCase):
    def test_unrepresentable_finite_scales_are_rejected(self):
        for alpha in (1e-300, 1e308):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    ValueError,
                    r"alpha\*\*2 \* \(n \+ kappa\) must be finite and positive",
                ):
                    MerweScaledSigmaPoints(
                        n=2,
                        alpha=alpha,
                        beta=2.0,
                        kappa=0.0,
                    )


if __name__ == "__main__":
    unittest.main()
