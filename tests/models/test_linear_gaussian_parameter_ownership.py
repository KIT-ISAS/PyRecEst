import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.models import (
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="JAX arrays are immutable",
)
class LinearGaussianParameterOwnershipTest(unittest.TestCase):
    def test_transition_model_copies_constructor_arrays(self):
        matrix = array([[1.0, 1.0], [0.0, 1.0]])
        noise_cov = array([[0.1, 0.0], [0.0, 0.2]])
        offset = array([0.5, -0.25])
        model = LinearGaussianTransitionModel(matrix, noise_cov, offset)

        matrix[0, 0] = 7.0
        noise_cov[0, 0] = 8.0
        offset[0] = 9.0

        npt.assert_allclose(
            to_numpy(model.system_matrix),
            [[1.0, 1.0], [0.0, 1.0]],
        )
        npt.assert_allclose(
            to_numpy(model.system_noise_cov),
            [[0.1, 0.0], [0.0, 0.2]],
        )
        npt.assert_allclose(to_numpy(model.sys_input), [0.5, -0.25])

    def test_measurement_model_copies_constructor_arrays(self):
        matrix = array([[1.0, 0.0]])
        noise_cov = array([[0.25]])
        model = LinearGaussianMeasurementModel(matrix, noise_cov)

        matrix[0, 0] = 7.0
        noise_cov[0, 0] = 8.0

        npt.assert_allclose(to_numpy(model.measurement_matrix), [[1.0, 0.0]])
        npt.assert_allclose(to_numpy(model.measurement_noise_cov), [[0.25]])


if __name__ == "__main__":
    unittest.main()
