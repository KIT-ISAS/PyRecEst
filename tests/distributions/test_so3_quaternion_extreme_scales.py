import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import so3_helpers


def test_normalize_quaternions_handles_extreme_finite_scales():
    backend_dtype = to_numpy(array([1.0])).dtype
    finfo = np.finfo(backend_dtype)
    quaternions = array(
        np.asarray(
            [
                [finfo.max, finfo.max / 2.0, 0.0, 0.0],
                [2.0 * finfo.tiny, finfo.tiny, 0.0, 0.0],
                [0.0, 0.0, 0.0, -finfo.max],
            ],
            dtype=backend_dtype,
        )
    )
    expected_direction = np.asarray(
        [2.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 0.0, 0.0],
        dtype=backend_dtype,
    )
    expected = np.stack(
        [
            expected_direction,
            expected_direction,
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=backend_dtype),
        ]
    )

    normalized = to_numpy(so3_helpers.normalize_quaternions(quaternions))

    assert np.all(np.isfinite(normalized))
    npt.assert_allclose(normalized, expected, rtol=1.0e-6, atol=1.0e-7)
