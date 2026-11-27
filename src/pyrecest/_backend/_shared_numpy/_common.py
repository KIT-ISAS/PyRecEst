from pyrecest._backend._dtype_utils import _np_box_binary_scalar as _box_binary_scalar  # noqa: F401
from pyrecest._backend._dtype_utils import _np_box_unary_scalar as _box_unary_scalar  # noqa: F401
from pyrecest._backend._dtype_utils import (
    _pre_add_default_dtype_by_casting,
    _pre_allow_complex_dtype,
    _pre_cast_fout_to_input_dtype,
    _pre_cast_out_from_dtype,
    _pre_cast_out_to_input_dtype,
    _pre_set_default_dtype,
)

from .._backend_config import np_atol as atol  # noqa: F401
from .._backend_config import np_rtol as rtol  # noqa: F401
from ._dispatch import numpy as _np

_DTYPES = {
    _np.dtype("bool"): 0,
    _np.dtype("int32"): 1,
    _np.dtype("int64"): 2,
    _np.dtype("float32"): 3,
    _np.dtype("float64"): 4,
    _np.dtype("complex64"): 5,
    _np.dtype("complex128"): 6,
}

_COMPLEX_DTYPES = [
    _np.complex64,
    _np.complex128,
]


def is_floating(x):
    return x.dtype.kind == "f"


def is_complex(x):
    return x.dtype.kind == "c"


def is_bool(x):
    return x.dtype.kind == "b"


def as_dtype(value):
    """Transform string representing dtype in dtype."""
    return _np.dtype(value)


def _dtype_as_str(dtype):
    return dtype.name


def _normalize_numpy_dtype(dtype):
    if dtype is None:
        return None
    try:
        return _np.dtype(dtype)
    except TypeError:
        return _np.dtype(str(dtype).split(".")[-1])


def cast(x, dtype):
    dtype = _normalize_numpy_dtype(dtype)
    if not hasattr(x, "astype"):
        return _np.asarray(x, dtype=dtype)
    return x.astype(dtype)


set_default_dtype = _pre_set_default_dtype(as_dtype)

_add_default_dtype_by_casting = _pre_add_default_dtype_by_casting(cast)
_cast_fout_to_input_dtype = _pre_cast_fout_to_input_dtype(cast, is_floating)
_cast_out_to_input_dtype = _pre_cast_out_to_input_dtype(
    cast, is_floating, is_complex, as_dtype, _dtype_as_str
)


_cast_out_from_dtype = _pre_cast_out_from_dtype(cast, is_floating, is_complex)
_allow_complex_dtype = _pre_allow_complex_dtype(cast, _COMPLEX_DTYPES)


def is_array(x):
    return x.__class__ is _np.ndarray


def to_ndarray(x, to_ndim, axis=0, dtype=None):
    x = _np.asarray(x, dtype=dtype)

    if x.ndim > to_ndim:
        raise ValueError("The ndim cannot be adapted properly.")

    while x.ndim < to_ndim:
        x = _np.expand_dims(x, axis=axis)

    return x


def _get_wider_dtype(tensor_list):
    if len(tensor_list) == 0:
        return None, True

    dtypes = [_np.dtype(x.dtype) for x in tensor_list]
    if all(dtype == dtypes[0] for dtype in dtypes[1:]):
        return dtypes[0], True

    try:
        common_dtype = _np.result_type(*dtypes)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "Cannot determine a common dtype for unsupported dtype(s): "
            f"{', '.join(str(dtype) for dtype in dtypes)}"
        ) from exc
    return common_dtype, False


def convert_to_wider_dtype(tensor_list):
    wider_dtype, same = _get_wider_dtype(tensor_list)
    if same:
        return tensor_list

    converted = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return converted


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        if not x:
            return False
        return _is_boolean(x[0])
    if isinstance(x, _np.ndarray):
        return x.dtype == bool
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if isinstance(x, _np.ndarray):
        return x.ndim > 0
    return False
