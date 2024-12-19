import ctypes

import numpy as np

from pycblas._loader import _mkl_lib
from pycblas.util import arrays, types

_MKL_INT = types.MKL_INT

caxpby = _mkl_lib.cblas_caxpby
caxpby.restype = None
daxpby = _mkl_lib.cblas_daxpby
daxpby.restype = None
saxpby = _mkl_lib.cblas_saxpby
saxpby.restype = None
zaxpby = _mkl_lib.cblas_zaxpby
zaxpby.restype = None

axpby_funcs = {
    np.float32: saxpby,
    np.float64: daxpby,
    np.complex64: caxpby,
    np.complex128: zaxpby,
}

saxpy_batch = _mkl_lib.cblas_saxpy_batch
saxpy_batch.restype = None
daxpy_batch = _mkl_lib.cblas_daxpy_batch
daxpy_batch.restype = None
caxpy_batch = _mkl_lib.caxpy_batch
caxpy_batch.restype = None
zaxpy_batch = _mkl_lib.cblas_zaxpy_batch
zaxpy_batch.restype = None

axpy_batch_funcs = {
    np.float32: saxpy_batch,
    np.float64: daxpy_batch,
    np.complex64: caxpy_batch,
    np.complex128: zaxpy_batch,
}

saxpy_batch_strided = _mkl_lib.cblas_saxpy_batch_strided
saxpy_batch_strided.restype = None
daxpy_batch_strided = _mkl_lib.cblas_daxpy_batch_strided
daxpy_batch_strided.restype = None
caxpy_batch_strided = _mkl_lib.cblas_caxpy_batch_strided
caxpy_batch_strided.restype = None
zaxpy_batch_strided = _mkl_lib.cblas_zaxpy_batch_strided
zaxpy_batch_strided.restype = None

axpy_batch_strided_funcs = {
    np.float32: saxpy_batch_strided,
    np.float64: daxpy_batch_strided,
    np.complex64: caxpy_batch_strided,
    np.complex128: zaxpy_batch_strided,
}

sgemm_batch = _mkl_lib.cblas_sgemm_batch
sgemm_batch.restype = None
dgemm_batch = _mkl_lib.cblas_dgemm_batch
dgemm_batch.restype = None
cgemm_batch = _mkl_lib.cblas_cgemm_batch
cgemm_batch.restype = None
zgemm_batch = _mkl_lib.cblas_zgemm_batch
zgemm_batch.restype = None

gemm_batch_funcs = {
    np.float32: sgemm_batch,
    np.float64: dgemm_batch,
    np.complex64: cgemm_batch,
    np.complex128: zgemm_batch,
}

sgemm_batch_strided = _mkl_lib.cblas_sgemm_batch_strided
sgemm_batch_strided.restype = None
dgemm_batch_strided = _mkl_lib.cblas_dgemm_batch_strided
dgemm_batch_strided.restype = None
cgemm_batch_strided = _mkl_lib.cblas_cgemm_batch_strided
cgemm_batch_strided.restype = None
zgemm_batch_strided = _mkl_lib.cblas_zgemm_batch_strided
zgemm_batch_strided.restype = None

gemm_batch_strided_funcs = {
    np.float32: sgemm_batch_strided,
    np.float64: dgemm_batch_strided,
    np.complex64: cgemm_batch_strided,
    np.complex128: zgemm_batch_strided,
}

sdgmm_batch = _mkl_lib.cblas_sdgmm_batch
sdgmm_batch.restype = None
ddgmm_batch = _mkl_lib.cblas_ddgmm_batch
ddgmm_batch.restype = None
cdgmm_batch = _mkl_lib.cblas_cdgmm_batch
cdgmm_batch.restype = None
zdgmm_batch = _mkl_lib.cblas_zdgmm_batch
zdgmm_batch.restype = None

dgmm_batch_funcs = {
    np.float32: sdgmm_batch,
    np.float64: ddgmm_batch,
    np.complex64: cdgmm_batch,
    np.complex128: zdgmm_batch,
}

sdgmm_batch_strided = _mkl_lib.cblas_sdgmm_batch_strided
sdgmm_batch_strided.restype = None
ddgmm_batch_strided = _mkl_lib.cblas_ddgmm_batch_strided
ddgmm_batch_strided.restype = None
cdgmm_batch_strided = _mkl_lib.cblas_cdgmm_batch_strided
cdgmm_batch_strided.restype = None
zdgmm_batch_strided = _mkl_lib.cblas_zdgmm_batch_strided
zdgmm_batch_strided.restype = None

dgmm_batch_strided_funcs = {
    np.float32: sdgmm_batch_strided,
    np.float64: ddgmm_batch_strided,
    np.complex64: cdgmm_batch_strided,
    np.complex128: zdgmm_batch_strided,
}

sgemv_batch = _mkl_lib.cblas_sgemv_batch
sgemv_batch.restype = None
dgemv_batch = _mkl_lib.cblas_dgemv_batch
dgemv_batch.restype = None
cgemv_batch = _mkl_lib.cblas_cgemv_batch
cgemv_batch.restype = None
zgemv_batch = _mkl_lib.cblas_zgemv_batch
zgemv_batch.restype = None


gemv_batch_funcs = {
    np.float32: sgemv_batch,
    np.float64: dgemv_batch,
    np.complex64: cgemv_batch,
    np.complex128: zgemv_batch,
}

sgemv_batch_strided = _mkl_lib.cblas_sgemv_batch_strided
sgemv_batch_strided.restype = None
dgemv_batch_strided = _mkl_lib.cblas_dgemv_batch_strided
dgemv_batch_strided.restype = None
cgemv_batch_strided = _mkl_lib.cblas_cgemv_batch_strided
cgemv_batch_strided.restype = None
zgemv_batch_strided = _mkl_lib.cblas_zgemv_batch_strided
zgemv_batch_strided.restype = None

gemv_batch_strided_funcs = {
    np.float32: sgemv_batch_strided,
    np.float64: dgemv_batch_strided,
    np.complex64: cgemv_batch_strided,
    np.complex128: zgemv_batch_strided,
}

sgemmt = _mkl_lib.cblas_sgemmt
sgemmt.restype = None
dgemmt = _mkl_lib.cblas_dgemmt
dgemmt.restype = None
cgemmt = _mkl_lib.cblas_cgemmt
cgemmt.restype = None
zgemmt = _mkl_lib.cblas_zgemmt
zgemmt.restype = None

gemmt_funcs = {
    np.float32: sgemmt,
    np.float64: dgemmt,
    np.complex64: cgemmt,
    np.complex128: zgemmt,
}

cgemm3m = _mkl_lib.cblas_cgemm3m
cgemm3m.restype = None
zgemm3m = _mkl_lib.cblas_zgemm3m
zgemm3m.restype = None

gemm3m_funcs = {
    np.complex64: cgemm3m,
    np.complex128: zgemm3m,
}

cgemm3m_batch = _mkl_lib.cblas_cgemm3m_batch
cgemm3m_batch.restype = None
zgemm3m_batch = _mkl_lib.cblas_zgemm3m_batch
zgemm3m_batch.restype = None

gemm3m_batch_funcs = {
    np.complex64: cgemm3m_batch,
    np.complex128: zgemm3m_batch,
}

# cgemm3m_batch_strided = _mkl_lib.cblas_cgemm3m_batch_strided
# cgemm3m_batch_strided.restype = None
# zgemm3m_batch_strided = _mkl_lib.cblas_zgemm3m_batch_strided
# zgemm3m_batch_strided.restype = None

# gemm3m_batch_strided_funcs = {
#     np.complex64: cgemm3m_batch_strided,
#     np.complex128: zgemm3m_batch_strided,
# }

strsm_batch = _mkl_lib.strsm_batch
strsm_batch.restype = None
dtrsm_batch = _mkl_lib.dtrsm_batch
dtrsm_batch.restype = None
ctrsm_batch = _mkl_lib.ctrsm_batch
ctrsm_batch.restype = None
ztrsm_batch = _mkl_lib.ztrsm_batch
ztrsm_batch.restype = None

trsm_batch_funcs = {
    np.float32: strsm_batch,
    np.float64: dtrsm_batch,
    np.complex64: ctrsm_batch,
    np.complex128: ztrsm_batch,
}

strsm_batch_strided = _mkl_lib.strsm_batch_strided
strsm_batch_strided.restype = None
dtrsm_batch_strided = _mkl_lib.dtrsm_batch_strided
dtrsm_batch_strided.restype = None
ctrsm_batch_strided = _mkl_lib.ctrsm_batch_strided
ctrsm_batch_strided.restype = None
ztrsm_batch_strided = _mkl_lib.ztrsm_batch_strided

trsm_batch_strided_funcs = {
    np.float32: strsm_batch_strided,
    np.float64: dtrsm_batch_strided,
    np.complex64: ctrsm_batch_strided,
    np.complex128: ztrsm_batch_strided,
}

mkl_simatcopy = _mkl_lib.mkl_simatcopy
mkl_simatcopy.restype = None
mkl_dimatcopy = _mkl_lib.mkl_dimatcopy
mkl_dimatcopy.restype = None
mkl_cimatcopy = _mkl_lib.mkl_cimatcopy
mkl_cimatcopy.restype = None
mkl_zimatcopy = _mkl_lib.mkl_zimatcopy
mkl_zimatcopy.restype = None

imatcopy_funcs = {
    np.float32: mkl_simatcopy,
    np.float64: mkl_dimatcopy,
    np.complex64: mkl_cimatcopy,
    np.complex128: mkl_zimatcopy,
}

mkl_simatcopy_batch = _mkl_lib.mkl_simatcopy_batch
mkl_simatcopy_batch.restype = None
mkl_dimatcopy_batch = _mkl_lib.mkl_dimatcopy_batch
mkl_dimatcopy_batch.restype = None
mkl_cimatcopy_batch = _mkl_lib.mkl_cimatcopy_batch
mkl_cimatcopy_batch.restype = None
mkl_zimatcopy_batch = _mkl_lib.mkl_zimatcopy_batch
mkl_zimatcopy_batch.restype = None

imatcopy_batch_funcs = {
    np.float32: mkl_simatcopy_batch,
    np.float64: mkl_dimatcopy_batch,
    np.complex64: mkl_cimatcopy_batch,
    np.complex128: mkl_zimatcopy_batch,
}

mkl_simatcopy_batch_strided = _mkl_lib.mkl_simatcopy_batch_strided
mkl_simatcopy_batch_strided.restype = None
mkl_dimatcopy_batch_strided = _mkl_lib.mkl_dimatcopy_batch_strided
mkl_dimatcopy_batch_strided.restype = None
mkl_cimatcopy_batch_strided = _mkl_lib.mkl_cimatcopy_batch_strided
mkl_cimatcopy_batch_strided.restype = None
mkl_zimatcopy_batch_strided = _mkl_lib.mkl_zimatcopy_batch_strided
mkl_zimatcopy_batch_strided.restype = None

imatcopy_batch_strided_funcs = {
    np.float32: mkl_simatcopy_batch_strided,
    np.float64: mkl_dimatcopy_batch_strided,
    np.complex64: mkl_cimatcopy_batch_strided,
    np.complex128: mkl_zimatcopy_batch_strided,
}

mkl_somatcopy = _mkl_lib.mkl_somatcopy
mkl_somatcopy.restype = None
mkl_domatcopy = _mkl_lib.mkl_domatcopy
mkl_domatcopy.restype = None
mkl_comatcopy = _mkl_lib.mkl_comatcopy
mkl_comatcopy.restype = None
mkl_zomatcopy = _mkl_lib.mkl_zomatcopy
mkl_zomatcopy.restype = None

omatcopy_funcs = {
    np.float32: mkl_somatcopy,
    np.float64: mkl_domatcopy,
    np.complex64: mkl_comatcopy,
    np.complex128: mkl_zomatcopy,
}

mkl_somatcopy_batch = _mkl_lib.mkl_somatcopy_batch
mkl_somatcopy_batch.restype = None
mkl_domatcopy_batch = _mkl_lib.mkl_domatcopy_batch
mkl_domatcopy_batch.restype = None
mkl_comatcopy_batch = _mkl_lib.mkl_comatcopy_batch
mkl_comatcopy_batch.restype = None
mkl_zomatcopy_batch = _mkl_lib.mkl_zomatcopy_batch
mkl_zomatcopy_batch.restype = None

omatcopy_batch_funcs = {
    np.float32: mkl_somatcopy_batch,
    np.float64: mkl_domatcopy_batch,
    np.complex64: mkl_comatcopy_batch,
    np.complex128: mkl_zomatcopy_batch,
}

mkl_somatcopy_batch_strided = _mkl_lib.mkl_somatcopy_batch_strided
mkl_somatcopy_batch_strided.restype = None
mkl_domatcopy_batch_strided = _mkl_lib.mkl_domatcopy_batch_strided
mkl_domatcopy_batch_strided.restype = None
mkl_comatcopy_batch_strided = _mkl_lib.mkl_comatcopy_batch_strided
mkl_comatcopy_batch_strided.restype = None
mkl_zomatcopy_batch_strided = _mkl_lib.mkl_zomatcopy_batch_strided
mkl_zomatcopy_batch_strided.restype = None

omatcopy_batch_strided_funcs = {
    np.float32: mkl_somatcopy_batch_strided,
    np.float64: mkl_domatcopy_batch_strided,
    np.complex64: mkl_comatcopy_batch_strided,
    np.complex128: mkl_zomatcopy_batch_strided,
}

mkl_somatadd = _mkl_lib.mkl_somatadd
mkl_somatadd.restype = None
mkl_domatadd = _mkl_lib.mkl_domatadd
mkl_domatadd.restype = None
mkl_comatadd = _mkl_lib.mkl_comatadd
mkl_comatadd.restype = None
mkl_zomatadd = _mkl_lib.mkl_zomatadd
mkl_zomatadd.restype = None

omatadd_funcs = {
    np.float32: mkl_somatadd,
    np.float64: mkl_domatadd,
    np.complex64: mkl_comatadd,
    np.complex128: mkl_zomatadd,
}


mkl_somatadd_batch_strided = _mkl_lib.mkl_somatadd_batch_strided
mkl_somatadd_batch_strided.restype = None
mkl_domatadd_batch_strided = _mkl_lib.mkl_domatadd_batch_strided
mkl_domatadd_batch_strided.restype = None
mkl_comatadd_batch_strided = _mkl_lib.mkl_comatadd_batch_strided
mkl_comatadd_batch_strided.restype = None
mkl_zomatadd_batch_strided = _mkl_lib.mkl_zomatadd_batch_strided
mkl_zomatadd_batch_strided.restype = None

omatadd_batch_strided_funcs = {
    np.float32: mkl_somatadd_batch_strided,
    np.float64: mkl_domatadd_batch_strided,
    np.complex64: mkl_comatadd_batch_strided,
    np.complex128: mkl_zomatadd_batch_strided,
}




def gemm_batch_strided(a, b, c, alpha=1.0, beta=0.0, conja=False, conjb=False):
    cstride, ldc, outorder = arrays.matrixstack_strides_order(c)
    nbatch, m, n, k = types.check_vmatmul_shapes(a, b, c)
    aflip, transa, astride, lda = arrays.get_matrixstack_args(outorder, a)
    bflip, transb, bstride, ldb = arrays.get_matrixstack_args(outorder, b)
    if conja:
        if transa == arrays.CblasNoTrans:
            raise ValueError("conja=True requires a and c to have different order")
        transa = arrays.CblasConjTrans
    if conjb:
        if transb == arrays.CblasNoTrans:
            raise ValueError("conjb=True requires b and c to have different order")
        transb = arrays.CblasConjTrans
    scalar_type = types.check_nd_types(a, b, c)
    gemm_batched_strided_func = gemm_batch_strided_funcs[scalar_type]
    gemm_batched_strided_func(
        ctypes.c_int(outorder),
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        _MKL_INT(m),
        _MKL_INT(n),
        _MKL_INT(k),
        types.scalar_arg_to_ctype(scalar_type, alpha),
        aflip.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        _MKL_INT(astride),
        bflip.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldb),
        _MKL_INT(bstride),
        types.scalar_arg_to_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
        _MKL_INT(cstride),
        _MKL_INT(nbatch),
    )

def dgmm(a, x, c, side='L'):
    ldc, outorder = arrays.leading_dimension_and_order(c)
    _, transa, lda = arrays.get_array_args(outorder, a)
    if transa == arrays.CblasTrans:
        raise ValueError("a and c must have the same order")
    if a.shape != c.shape:
        raise ValueError("a and c must have the same shape")
    assert x.ndim == 1
    incx = arrays.get_elem_strides(x)[0]
    scalar_type = types.check_nd_types(a, x, c)
    dgmm_func = dgmm_batch_strided_funcs[scalar_type]
    dgmm_func(
        ctypes.c_int(outorder),
        ctypes.c_int(arrays.get_cblas_side(side)),
        _MKL_INT(a.shape[0]),
        _MKL_INT(a.shape[1]),
        a.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        _MKL_INT(a.size),
        x.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(incx),
        _MKL_INT(0),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
        _MKL_INT(c.size),
        _MKL_INT(1),
    )