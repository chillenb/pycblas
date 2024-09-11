import ctypes

import numpy as np

from pymkl._loader import _mkl_lib
from pymkl.util import types

_MKL_INT = types.MKL_INT

sasum = _mkl_lib.cblas_sasum
sasum.restype = ctypes.c_float
dasum = _mkl_lib.cblas_dasum
dasum.restype = ctypes.c_double
scasum = _mkl_lib.cblas_scasum
scasum.restype = ctypes.c_float
dzasum = _mkl_lib.cblas_dzasum
dzasum.restype = ctypes.c_double


def asum(x):
    """Compute the sum of the absolute values of a vector

    Parameters
    ----------
    x : numpy.ndarray
        Input array

    Returns
    -------
    float
        Sum of the absolute values of the input array
    """
    scalar_type, size = types.check_vec_types(x)
    elem_size = x.dtype.itemsize
    incx = x.strides[0] // elem_size
    if scalar_type == np.float32:
        return sasum(_MKL_INT(size), x.ctypes.data_as(types.c_float_p), _MKL_INT(incx))
    if scalar_type == np.float64:
        return dasum(_MKL_INT(size), x.ctypes.data_as(types.c_double_p), _MKL_INT(incx))
    if scalar_type == np.complex64:
        return scasum(_MKL_INT(size), x.ctypes.data_as(ctypes.c_void_p), _MKL_INT(incx))
    if scalar_type == np.complex128:
        return dzasum(_MKL_INT(size), x.ctypes.data_as(ctypes.c_void_p), _MKL_INT(incx))
    msg = f"Unsupported type: {scalar_type}"
    raise ValueError(msg)
