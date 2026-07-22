import numpy as np
import pytest
from pyrecest._backend.numpy import linalg


@pytest.mark.parametrize(
    ("matrix_function", "args"),
    [
        pytest.param(linalg.logm, (), id="logm"),
        pytest.param(linalg.sqrtm, (), id="sqrtm"),
        pytest.param(
            linalg.fractional_matrix_power,
            (0.5,),
            id="fractional_matrix_power",
        ),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        pytest.param(np.float32, np.float32, id="float32"),
        pytest.param(np.complex64, np.complex64, id="complex64"),
        pytest.param(np.int64, np.float64, id="integer-promoted"),
    ],
)
def test_empty_matrix_function_batches_preserve_shape_and_dtype(
    matrix_function, args, dtype, expected_dtype
):
    matrices = np.empty((2, 0, 3, 3), dtype=dtype)

    result = matrix_function(matrices, *args)

    assert result.shape == matrices.shape
    assert result.dtype == np.dtype(expected_dtype)


@pytest.mark.parametrize(
    ("matrix_function", "args"),
    [
        pytest.param(linalg.logm, (), id="logm"),
        pytest.param(linalg.sqrtm, (), id="sqrtm"),
        pytest.param(
            linalg.fractional_matrix_power,
            (0.5,),
            id="fractional_matrix_power",
        ),
    ],
)
def test_empty_rectangular_matrix_function_batches_remain_invalid(
    matrix_function, args
):
    matrices = np.empty((0, 3, 2))

    with pytest.raises(ValueError):
        matrix_function(matrices, *args)
