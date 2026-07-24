from __future__ import annotations

import numpy as np
import numpy.testing as npt
from pyrecest.tracking import ellipse_extent_matrix, extent_matrix_from_shape


def test_extent_matrix_preserves_negative_axis_magnitudes() -> None:
    orientation = 0.37
    signed_axes = np.array([-3.0, 1.5])
    positive_axes = np.abs(signed_axes)
    expected_extent = ellipse_extent_matrix(orientation, positive_axes)

    npt.assert_allclose(
        ellipse_extent_matrix(orientation, signed_axes),
        expected_extent,
        atol=1e-12,
    )
    npt.assert_allclose(
        extent_matrix_from_shape(
            np.array([orientation, signed_axes[0], signed_axes[1]])
        ),
        expected_extent,
        atol=1e-12,
    )
