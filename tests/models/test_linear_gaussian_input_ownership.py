import numpy as np

from pyrecest.models import (
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


def test_transition_model_copies_mutable_constructor_arrays():
    matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    noise_cov = np.diag([0.1, 0.2])
    offset = np.array([0.5, -0.5])
    expected_matrix = matrix.copy()
    expected_noise_cov = noise_cov.copy()
    expected_offset = offset.copy()

    model = LinearGaussianTransitionModel(matrix, noise_cov, offset)

    matrix[:] = -10.0
    noise_cov[:] = 99.0
    offset[:] = 42.0

    np.testing.assert_allclose(model.matrix, expected_matrix)
    np.testing.assert_allclose(model.noise_cov, expected_noise_cov)
    np.testing.assert_allclose(model.offset, expected_offset)
    np.testing.assert_allclose(
        model.predict_mean(np.array([1.0, 2.0])),
        expected_matrix @ np.array([1.0, 2.0]) + expected_offset,
    )


def test_measurement_model_copies_mutable_constructor_arrays():
    matrix = np.array([[1.0, 0.0], [0.5, 2.0]])
    noise_cov = np.diag([0.3, 0.4])
    expected_matrix = matrix.copy()
    expected_noise_cov = noise_cov.copy()

    model = LinearGaussianMeasurementModel(matrix, noise_cov)

    matrix[:] = -10.0
    noise_cov[:] = 99.0

    np.testing.assert_allclose(model.matrix, expected_matrix)
    np.testing.assert_allclose(model.noise_cov, expected_noise_cov)
    np.testing.assert_allclose(
        model.predict_mean(np.array([2.0, 3.0])),
        expected_matrix @ np.array([2.0, 3.0]),
    )
