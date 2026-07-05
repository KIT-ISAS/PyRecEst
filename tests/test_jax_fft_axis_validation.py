import numpy as np
import pytest


def test_jax_real_fft_rejects_non_scalar_singleton_array_axes():
    pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax import fft

    samples = np.arange(6.0).reshape(2, 3)
    spectrum = fft.rfft(samples, axis=1)

    for axis in (np.array([1]), np.array([[1]])):
        with pytest.raises(TypeError):
            fft.rfft(samples, axis=axis)
        with pytest.raises(TypeError):
            fft.irfft(spectrum, n=samples.shape[1], axis=axis)
