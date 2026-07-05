import numpy as np
import pyrecest.backend as backend
import pytest


def _as_numpy(value):
    return np.asarray(backend.to_numpy(value))


def test_jax_real_fft_accepts_numpy_scalar_array_axes():
    if backend.__backend_name__ != "jax":
        pytest.skip("JAX-specific FFT backend contract")

    samples = np.arange(6.0).reshape(2, 3)
    backend_samples = backend.asarray(samples)
    expected_spectrum = _as_numpy(backend.fft.rfft(backend_samples, axis=1))

    for axis in (np.array(1), np.int64(1)):
        actual_spectrum = _as_numpy(backend.fft.rfft(backend_samples, axis=axis))
        assert np.allclose(actual_spectrum, expected_spectrum)

        actual_roundtrip = _as_numpy(
            backend.fft.irfft(actual_spectrum, n=samples.shape[1], axis=axis)
        )
        assert np.allclose(actual_roundtrip, samples)
