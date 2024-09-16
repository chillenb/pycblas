import ctypes

import numpy as np

from pymkl._loader import _mkl_lib
from pymkl.util import arrays, types

_MKL_INT = types.MKL_INT

sgemm = _mkl_lib.cblas_sgemm
sgemm.restype = None
dgemm = _mkl_lib.cblas_dgemm
dgemm.restype = None
cgemm = _mkl_lib.cblas_cgemm
cgemm.restype = None
zgemm = _mkl_lib.cblas_zgemm
zgemm.restype = None

gemm_funcs = { np.float32: sgemm, np.float64: dgemm, np.complex64: cgemm, np.complex128: zgemm }

chemm = _mkl_lib.cblas_chemm
chemm.restype = None
zhemm = _mkl_lib.cblas_zhemm
zhemm.restype = None

hemm_funcs = { np.complex64: chemm, np.complex128: zhemm }

cherk = _mkl_lib.cblas_cherk
cherk.restype = None
zherk = _mkl_lib.cblas_zherk
zherk.restype = None

herk_funcs = { np.complex64: cherk, np.complex128: zherk }

cher2k = _mkl_lib.cblas_cher2k
cher2k.restype = None
zher2k = _mkl_lib.cblas_zher2k
zher2k.restype = None

her2k_funcs = { np.complex64: cher2k, np.complex128: zher2k }

ssymm = _mkl_lib.cblas_ssymm
ssymm.restype = None
dsymm = _mkl_lib.cblas_dsymm
dsymm.restype = None
csymm = _mkl_lib.cblas_csymm
csymm.restype = None
zsymm = _mkl_lib.cblas_zsymm
zsymm.restype = None

symm_funcs = { np.float32: ssymm, np.float64: dsymm, np.complex64: csymm, np.complex128: zsymm }

ssyrk = _mkl_lib.cblas_ssyrk
ssyrk.restype = None
dsyrk = _mkl_lib.cblas_dsyrk
dsyrk.restype = None
csyrk = _mkl_lib.cblas_csyrk
csyrk.restype = None
zsyrk = _mkl_lib.cblas_zsyrk
zsyrk.restype = None

syrk_funcs = { np.float32: ssyrk, np.float64: dsyrk, np.complex64: csyrk, np.complex128: zsyrk }


cherk = _mkl_lib.cblas_cherk
cherk.restype = None
zherk = _mkl_lib.cblas_zherk
zherk.restype = None

herk_funcs = { np.complex64: cherk, np.complex128: zherk }

ssyr2k = _mkl_lib.cblas_ssyr2k
ssyr2k.restype = None
dsyr2k = _mkl_lib.cblas_dsyr2k
dsyr2k.restype = None
csyr2k = _mkl_lib.cblas_csyr2k
csyr2k.restype = None
zsyr2k = _mkl_lib.cblas_zsyr2k
zsyr2k.restype = None

syr2k_funcs = { np.float32: ssyr2k, np.float64: dsyr2k, np.complex64: csyr2k, np.complex128: zsyr2k }

cher2k = _mkl_lib.cblas_cher2k
cher2k.restype = None
zher2k = _mkl_lib.cblas_zher2k
zher2k.restype = None

her2k_funcs = { np.complex64: cher2k, np.complex128: zher2k }

strmm = _mkl_lib.cblas_strmm
strmm.restype = None
dtrmm = _mkl_lib.cblas_dtrmm
dtrmm.restype = None
ctrmm = _mkl_lib.cblas_ctrmm
ctrmm.restype = None
ztrmm = _mkl_lib.cblas_ztrmm
ztrmm.restype = None

trmm_funcs = { np.float32: strmm, np.float64: dtrmm, np.complex64: ctrmm, np.complex128: ztrmm }

strsm = _mkl_lib.cblas_strsm
strsm.restype = None
dtrsm = _mkl_lib.cblas_dtrsm
dtrsm.restype = None
ctrsm = _mkl_lib.cblas_ctrsm
ctrsm.restype = None
ztrsm = _mkl_lib.cblas_ztrsm
ztrsm.restype = None

trsm_funcs = { np.float32: strsm, np.float64: dtrsm, np.complex64: ctrsm, np.complex128: ztrsm }

def gemm(a, b, c, alpha=1.0, beta=0.0):
    ldc, outorder = arrays.leading_dimension_and_order(c)
    m, n, k = types.check_matmul_shapes(a, b, c)
    aflip, transa, lda = arrays.get_array_args(outorder, a)
    bflip, transb, ldb = arrays.get_array_args(outorder, b)
    scalar_type = types.check_nd_types(a, b, c)
    gemm_func = gemm_funcs[scalar_type]
    gemm_func(
        ctypes.c_int(outorder),
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        _MKL_INT(m),
        _MKL_INT(n),
        _MKL_INT(k),
        types.scalar_arg_to_ctype(scalar_type, alpha),
        aflip.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        bflip.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldb),
        types.scalar_arg_to_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
    )

def hemm(side, uplo, a, b, c, alpha=1.0, beta=0.0):
    ldc, outorder = arrays.leading_dimension_and_order(c)
    lda, a_order = arrays.leading_dimension_and_order(a)
    ldb, b_order = arrays.leading_dimension_and_order(b)
    if a_order != outorder or b_order != outorder:
        raise ValueError("All matrices must have the same order for hemm")
    cblas_side = arrays.get_cblas_side(side)
    cblas_uplo = arrays.get_cblas_uplo(uplo)
    if cblas_side == arrays.CblasLeft:
        m, n, _ = types.check_matmul_shapes(a, b, c)
    else:
        m, n, _ = types.check_matmul_shapes(b, a, c)
    scalar_type = types.check_nd_types(a, b, c)
    hemm_func = hemm_funcs[scalar_type]
    hemm_func(
        ctypes.c_int(outorder),
        ctypes.c_int(cblas_side),
        ctypes.c_int(cblas_uplo),
        _MKL_INT(m),
        _MKL_INT(n),
        types.scalar_arg_to_ctype(scalar_type, alpha),
        a.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        b.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldb),
        types.scalar_arg_to_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
    )

def symm(side, uplo, a, b, c, alpha=1.0, beta=0.0):
    ldc, outorder = arrays.leading_dimension_and_order(c)
    lda, a_order = arrays.leading_dimension_and_order(a)
    ldb, b_order = arrays.leading_dimension_and_order(b)
    if a_order != outorder or b_order != outorder:
        raise ValueError("All matrices must have the same order for symm")
    cblas_side = arrays.get_cblas_side(side)
    cblas_uplo = arrays.get_cblas_uplo(uplo)
    if cblas_side == arrays.CblasLeft:
        m, n, _ = types.check_matmul_shapes(a, b, c)
    else:
        m, n, _ = types.check_matmul_shapes(b, a, c)
    scalar_type = types.check_nd_types(a, b, c)
    symm_func = symm_funcs[scalar_type]
    symm_func(
        ctypes.c_int(outorder),
        ctypes.c_int(cblas_side),
        ctypes.c_int(cblas_uplo),
        _MKL_INT(m),
        _MKL_INT(n),
        types.scalar_arg_to_ctype(scalar_type, alpha),
        a.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        b.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldb),
        types.scalar_arg_to_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
    )

def syrk(uplo, a, c, alpha=1.0, beta=0.0):
    """Symmetric rank-k update.
    c := alpha * a @ a.T + beta * c

    Parameters
    ----------
    uplo : char
        'U' or 'L'. Specifies whether the upper or lower triangular part of the array c is used.
    a : array_like
        n by k array
    c : array_like
        n by n array
    alpha : scalar, optional
        by default 1.0
    beta : scalar, optional
        by default 0.0
    """
    ldc, outorder = arrays.leading_dimension_and_order(c)
    aflip, transa, lda = arrays.get_array_args(outorder, a)
    cblas_uplo = arrays.get_cblas_uplo(uplo)
    n = c.shape[0]
    k = a.shape[1]
    if c.shape != (n, n):
        raise ValueError("c must be square")
    if a.shape[0] != n:
        raise ValueError("a must have the same number of rows as c")
    scalar_type = types.check_nd_types(a, c)
    syrk_func = syrk_funcs[scalar_type]
    syrk_func(
        ctypes.c_int(outorder),
        ctypes.c_int(cblas_uplo),
        ctypes.c_int(transa),
        _MKL_INT(n),
        _MKL_INT(k),
        types.scalar_arg_to_ctype(scalar_type, alpha),
        aflip.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        types.scalar_arg_to_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
    )

def herk(uplo, conjtrans, a, c, alpha=1.0, beta=0.0):
    """Hermitian rank-k update.
    c := alpha * a @ a.conj().T + beta * c

    Parameters
    ----------
    uplo : char
        'U' or 'L'. Specifies whether the upper or lower triangular part of the array c is used.
    conjtrans:
        'N' or 'C'. Specifies whether to use a or a.conj().T
    a : array_like
        n by k array
    c : array_like
        n by n array
    alpha : float, optional
        by default 1.0
    beta : float, optional
        by default 0.0
    """
    ldc, outorder = arrays.leading_dimension_and_order(c)
    lda, a_order = arrays.leading_dimension_and_order(a)
    trans = arrays.get_cblas_conjtrans(conjtrans)
    if a_order != outorder:
        raise ValueError("a and c must have the same order")
    if trans != arrays.CblasNoTrans:
        k, _ = a.shape
    else:
        _, k = a.shape
    cblas_uplo = arrays.get_cblas_uplo(uplo)
    n = c.shape[0]
    if c.shape != (n, n):
        raise ValueError("c must be square")
    scalar_type = types.check_nd_types(a, c)
    herk_func = herk_funcs[scalar_type]
    herk_func(
        ctypes.c_int(outorder),
        ctypes.c_int(cblas_uplo),
        ctypes.c_int(trans),
        _MKL_INT(n),
        _MKL_INT(k),
        types.scalar_arg_to_real_ctype(scalar_type, alpha),
        a.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        types.scalar_arg_to_real_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
    )

def syr2k(uplo, a, b, c, alpha=1.0, beta=0.0):
    """Symmetric rank-2k update.
    c := alpha * a @ b.T + alpha * b @ a.T + beta * c

    Parameters
    ----------
    uplo : char
        'U' or 'L'. Specifies whether the upper or lower triangular part of the array c is used.
    a : array_like
        n by k array
    b : array_like
        n by k array
    c : array_like
        n by n array
    alpha : scalar, optional
        by default 1.0
    beta : scalar, optional
        by default 0.0
    """
    ldc, outorder = arrays.leading_dimension_and_order(c)
    aflip, transa, lda = arrays.get_array_args(outorder, a)
    bflip, transb, ldb = arrays.get_array_args(outorder, b)
    if transa != transb:
        raise ValueError("a and b must be both F-contig or C-contig")
    cblas_uplo = arrays.get_cblas_uplo(uplo)
    n = c.shape[0]
    k = a.shape[1]
    if c.shape != (n, n):
        raise ValueError("c must be square")
    if a.shape[0] != n or b.shape[0] != n:
        raise ValueError("a and b must have the same number of rows as c")
    if b.shape[1] != k:
        raise ValueError("a and b must have the same number of columns")
    scalar_type = types.check_nd_types(a, b, c)
    syr2k_func = syr2k_funcs[scalar_type]
    syr2k_func(
        ctypes.c_int(outorder),
        ctypes.c_int(cblas_uplo),
        ctypes.c_int(transa),
        _MKL_INT(n),
        _MKL_INT(k),
        types.scalar_arg_to_ctype(scalar_type, alpha),
        aflip.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        bflip.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldb),
        types.scalar_arg_to_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
    )

def her2k(uplo, conjtrans, a, b, c, alpha=1.0, beta=0.0):
    """Hermitian rank-2k update.
    c := alpha * a @ b.conj().T + alpha * b @ a.conj().T + beta * c

    Parameters
    ----------
    uplo : char
        'U' or 'L'. Specifies whether the upper or lower triangular part of the array c is used.
    conjtrans:
        'N' or 'C'. Specifies whether to use a, b or a.conj().T, b.conj().T
    a : array_like
        n by k array
    b : array_like
        n by k array
    c : array_like
        n by n array
    alpha : scalar, optional
        by default 1.0
    beta : float, optional
        by default 0.0
    """
    ldc, outorder = arrays.leading_dimension_and_order(c)
    lda, a_order = arrays.leading_dimension_and_order(a)
    ldb, b_order = arrays.leading_dimension_and_order(b)
    trans = arrays.get_cblas_conjtrans(conjtrans)
    if a_order != outorder or a_order != b_order:
        raise ValueError("a, b, and c must have the same order")
    if trans != arrays.CblasNoTrans:
        ka, na = a.shape
        kb, nb = b.shape
    else:
        na, ka = a.shape
        nb, kb = b.shape
    if na != nb or ka != kb:
        raise ValueError("a and b must have the same shape")
    cblas_uplo = arrays.get_cblas_uplo(uplo)
    n = c.shape[0]
    if c.shape != (n, n):
        raise ValueError("c must be square")
    if na != n:
        raise ValueError("a and c must have the same number of rows")
    scalar_type = types.check_nd_types(a, b, c)
    her2k_func = her2k_funcs[scalar_type]
    her2k_func(
        ctypes.c_int(outorder),
        ctypes.c_int(cblas_uplo),
        ctypes.c_int(trans),
        _MKL_INT(n),
        _MKL_INT(ka),
        types.scalar_arg_to_ctype(scalar_type, alpha),
        a.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(lda),
        b.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldb),
        types.scalar_arg_to_real_ctype(scalar_type, beta),
        c.ctypes.data_as(ctypes.c_void_p),
        _MKL_INT(ldc),
    )
