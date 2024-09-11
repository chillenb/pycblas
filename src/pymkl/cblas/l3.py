from pymkl._loader import _mkl_lib
from pymkl.util import types, arrays
import numpy as np
import ctypes

_MKL_INT = types._MKL_INT

sgemm = _mkl_lib.cblas_sgemm
sgemm.restype = None
dgemm = _mkl_lib.cblas_dgemm
dgemm.restype = None
cgemm = _mkl_lib.cblas_cgemm
cgemm.restype = None
zgemm = _mkl_lib.cblas_zgemm
zgemm.restype = None



def gemm(a, b, c, alpha=1.0, beta=0.0):
    ldc, outorder = arrays.leading_dimension_and_order(c)
    m, n, k = types.check_matmul_shapes(a, b, c)
    aflip, transa, lda = arrays.get_array_args(outorder, a)
    bflip, transb, ldb = arrays.get_array_args(outorder, b)
    scalar_type = types.check_nd_types(a, b, c)
    if scalar_type == np.float32:
        sgemm(ctypes.c_int(outorder),
                  ctypes.c_int(transa),
                  ctypes.c_int(transb),
                  _MKL_INT(m),
                  _MKL_INT(n),
                  _MKL_INT(k),
                  ctypes.c_float(alpha),
                  aflip.ctypes.data_as(types.c_float_p),
                  _MKL_INT(lda),
                  bflip.ctypes.data_as(types.c_float_p),
                  _MKL_INT(ldb),
                  ctypes.c_float(beta),
                  c.ctypes.data_as(types.c_float_p),
                  _MKL_INT(ldc),
        )
    elif scalar_type == np.float64:
        dgemm(ctypes.c_int(outorder),
                  ctypes.c_int(transa),
                  ctypes.c_int(transb),
                  _MKL_INT(m),
                  _MKL_INT(n),
                  _MKL_INT(k),
                  ctypes.c_double(alpha),
                  aflip.ctypes.data_as(types.c_double_p),
                  _MKL_INT(lda),
                  bflip.ctypes.data_as(types.c_double_p),
                  _MKL_INT(ldb),
                  ctypes.c_double(beta),
                  c.ctypes.data_as(types.c_double_p),
                  _MKL_INT(ldc),
        )
    elif scalar_type == np.complex64:
        cgemm(ctypes.c_int(outorder),
                  ctypes.c_int(transa),
                  ctypes.c_int(transb),
                  _MKL_INT(m),
                  _MKL_INT(n),
                  _MKL_INT(k),
                  (ctypes.c_float*2)(alpha.real, alpha.imag),
                  aflip.ctypes.data_as(types.c_float_p),
                  _MKL_INT(lda),
                  bflip.ctypes.data_as(types.c_float_p),
                  _MKL_INT(ldb),
                  (ctypes.c_float*2)(beta.real, beta.imag),
                  c.ctypes.data_as(types.c_float_p),
                  _MKL_INT(ldc),
        )
    elif scalar_type == np.complex128:
        zgemm(ctypes.c_int(outorder),
                  ctypes.c_int(transa),
                  ctypes.c_int(transb),
                  _MKL_INT(m),
                  _MKL_INT(n),
                  _MKL_INT(k),
                  (ctypes.c_double*2)(alpha.real, alpha.imag),
                  aflip.ctypes.data_as(ctypes.c_void_p),
                  _MKL_INT(lda),
                  bflip.ctypes.data_as(ctypes.c_void_p),
                  _MKL_INT(ldb),
                  (ctypes.c_double*2)(beta.real, beta.imag),
                  c.ctypes.data_as(ctypes.c_void_p),
                  _MKL_INT(ldc),
        )
    else:
        msg = f"Unsupported type: {scalar_type}"
        raise ValueError(msg)
