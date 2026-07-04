import numpy as np
import pytest
from pyrecest._backend.numpy import random


@pytest.mark.parametrize(
    "dtype",
    [
        bool,
        np.bool_,
        np.dtype("bool"),
        float,
        np.float64,
        np.dtype("float32"),
        object,
    ],
)
def test_randint_rejects_non_integer_dtypes(dtype):
    with pytest.raises(TypeError, match="integer dtype"):
        random.randint(0, 2, size=3, dtype=dtype)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (int, np.dtype(int)),
        (np.int32, np.dtype("int32")),
        (np.dtype("uint32"), np.dtype("uint32")),
    ],
)
def test_randint_accepts_integer_dtype_aliases(dtype, expected_dtype):
    samples = random.randint(0, 2, size=3, dtype=dtype)

    assert samples.shape == (3,)
    assert samples.dtype == expected_dtype
    assert np.all(samples >= 0)
    assert np.all(samples < 2)
