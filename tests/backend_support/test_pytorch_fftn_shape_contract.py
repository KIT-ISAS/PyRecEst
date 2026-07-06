import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("torch")

import pyrecest._backend.pytorch.fft as pytorch_fft  # noqa: E402


@pytest.mark.backend_portable
def test_raw_pytorch_fftn_accepts_numpy_shape_array_keyword():
    vector = np.arange(4.0)
    shape = np.array([4], dtype=np.int64)

    spectrum = pytorch_fft.fftn(vector.tolist(), s=shape)

    npt.assert_allclose(spectrum.numpy(), np.fft.fftn(vector, s=shape))


@pytest.mark.backend_portable
def test_raw_pytorch_ifftn_accepts_numpy_shape_array_positional():
    vector = np.arange(4.0)
    shape = np.array([4], dtype=np.int64)
    spectrum = np.fft.fftn(vector, s=shape)

    reconstructed = pytorch_fft.ifftn(spectrum.tolist(), shape)

    npt.assert_allclose(reconstructed.numpy(), np.fft.ifftn(spectrum, s=shape))
