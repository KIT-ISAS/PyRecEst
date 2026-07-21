import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pyrecest._backend.pytorch.fft as pytorch_fft  # noqa: E402


@pytest.mark.backend_portable
@pytest.mark.parametrize("fft_func", [pytorch_fft.rfft, pytorch_fft.irfft])
@pytest.mark.parametrize(
    "length",
    [
        True,
        np.bool_(True),
        np.array(True),
        torch.tensor(True),
        torch.tensor([True]),
    ],
)
def test_raw_pytorch_real_fft_rejects_boolean_lengths(fft_func, length):
    with pytest.raises(TypeError, match="n must be an integer length, not boolean"):
        fft_func([1.0, 2.0, 3.0, 4.0], n=length)


@pytest.mark.backend_portable
@pytest.mark.parametrize("fft_func", [pytorch_fft.rfft, pytorch_fft.irfft])
def test_raw_pytorch_real_fft_rejects_positional_boolean_tensor_length(fft_func):
    with pytest.raises(TypeError, match="n must be an integer length, not boolean"):
        fft_func([1.0, 2.0, 3.0, 4.0], torch.tensor(True))


@pytest.mark.backend_portable
def test_raw_pytorch_real_fft_preserves_singleton_integer_tensor_length():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])

    actual = pytorch_fft.rfft(values, n=torch.tensor([4]))
    expected = torch.fft.rfft(values, n=4)

    torch.testing.assert_close(actual, expected)
