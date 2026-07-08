import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import estimate_transform


class TestEstimateTransformWeightValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_nonfinite_weights(self):
        source = array([[0.0, 0.0], [1.0, 0.0]])
        target = source + array([1.0, -1.0])

        for bad_weight in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_weight=bad_weight):
                with self.assertRaisesRegex(ValueError, "finite"):
                    estimate_transform(
                        source,
                        target,
                        model="translation",
                        weights=array([1.0, bad_weight]),
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_insufficient_positive_weight_support(self):
        affine_source = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        affine_target = affine_source + array([1.0, -1.0])

        with self.assertRaisesRegex(ValueError, "positive-weight matched points"):
            estimate_transform(
                affine_source,
                affine_target,
                model="affine",
                weights=array([1.0, 0.0, 0.0]),
            )

        rigid_source = array([[0.0, 0.0], [1.0, 0.0]])
        rigid_target = rigid_source + array([0.5, -0.5])
        with self.assertRaisesRegex(ValueError, "positive-weight matched points"):
            estimate_transform(
                rigid_source,
                rigid_target,
                model="rigid",
                weights=array([1.0, 0.0]),
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_accepts_zero_weights_when_support_is_identifiable(self):
        source = array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, -3.0]],
        )
        true_matrix = array([[1.2, -0.3], [0.4, 0.9]])
        true_offset = array([2.0, -1.5])
        target = (true_matrix @ source.T).T + true_offset

        estimated = estimate_transform(
            source,
            target,
            model="affine",
            weights=array([1.0, 1.0, 1.0, 0.0]),
        )

        npt.assert_allclose(estimated.matrix, true_matrix, atol=1e-10)
        npt.assert_allclose(estimated.offset, true_offset, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
