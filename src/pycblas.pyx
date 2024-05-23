# cython: language_level=3

import numpy as np
from cblas cimport *

ctypedef double complex dcomplex
ctypedef float complex fcomplex

ctypedef fused scalar_t:
    double
    float
    dcomplex
    fcomplex

ctypedef fused real_t:
    double
    float

ctypedef fused cplx_t:
    dcomplex
    fcomplex

def cblas_asum(scalar_t[:] x):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]

    if scalar_t is double:
        return cblas_dasum(N, &x[0], incx)
    elif scalar_t is float:
        return cblas_sasum(N, &x[0], incx)
    elif scalar_t is dcomplex:
        return cblas_dzasum(N, <void*> &x[0], incx)
    elif scalar_t is fcomplex:
        return cblas_scasum(N, <void*> &x[0], incx)

def cblas_axpy(scalar_t[:] x, scalar_t[:] y, scalar_t a):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]
    if N != y.shape[0]:
        raise ValueError("x and y must have the same length")

    if scalar_t is double:
        cblas_daxpy(N, a, &x[0], incx, &y[0], incy)
    elif scalar_t is float:
        cblas_saxpy(N, a, &x[0], incx, &y[0], incy)
    elif scalar_t is dcomplex:
        cblas_zaxpy(N, <void*> &a, <void*> &x[0], incx, <void*> &y[0], incy)
    elif scalar_t is fcomplex:
        cblas_caxpy(N, <void*> &a, <void*> &x[0], incx, <void*> &y[0], incy)

def cblas_copy(scalar_t[:] x, scalar_t[:] y):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]
    if N != y.shape[0]:
        raise ValueError("x and y must have the same length")

    if scalar_t is double:
        cblas_dcopy(N, &x[0], incx, &y[0], incy)
    elif scalar_t is float:
        cblas_scopy(N, &x[0], incx, &y[0], incy)
    elif scalar_t is dcomplex:
        cblas_zcopy(N, <void*> &x[0], incx, <void*> &y[0], incy)
    elif scalar_t is fcomplex:
        cblas_ccopy(N, <void*> &x[0], incx, <void*> &y[0], incy)


def cblas_dot(real_t[:] x, real_t[:] y):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(real_t)
    cdef CBLAS_INT incy = y.strides[0] // sizeof(real_t)
    cdef CBLAS_INT N = x.shape[0]
    if N != y.shape[0]:
        raise ValueError("x and y must have the same length")

    if real_t is double:
        return cblas_ddot(N, &x[0], incx, &y[0], incy)
    elif real_t is float:
        return cblas_sdot(N, &x[0], incx, &y[0], incy)

def cblas_dotc(scalar_t[:] x, scalar_t[:] y):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]
    cdef scalar_t result
    if N != y.shape[0]:
        raise ValueError("x and y must have the same length")

    if scalar_t is double:
        return cblas_ddot(N, &x[0], incx, &y[0], incy)
    elif scalar_t is float:
        return cblas_sdot(N, &x[0], incx, &y[0], incy)
    elif scalar_t is dcomplex:
        cblas_zdotc_sub(N, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &result)
    elif scalar_t is fcomplex:
        cblas_cdotc_sub(N, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &result)
    return result

def cblas_dotu(scalar_t[:] x, scalar_t[:] y):
    
    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]
    cdef scalar_t result
    if N != y.shape[0]:
        raise ValueError("x and y must have the same length")

    if scalar_t is double:
        return cblas_ddot(N, &x[0], incx, &y[0], incy)
    elif scalar_t is float:
        return cblas_sdot(N, &x[0], incx, &y[0], incy)
    elif scalar_t is dcomplex:
        cblas_zdotu_sub(N, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &result)
    elif scalar_t is fcomplex:
        cblas_cdotu_sub(N, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &result)
    return result

def cblas_nrm2(scalar_t[:] x):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]

    if scalar_t is double:
        return cblas_dnrm2(N, &x[0], incx)
    elif scalar_t is float:
        return cblas_snrm2(N, &x[0], incx)
    elif scalar_t is dcomplex:
        return cblas_dznrm2(N, <void*> &x[0], incx)
    elif scalar_t is fcomplex:
        return cblas_scnrm2(N, <void*> &x[0], incx)

def cblas_rot(real_t[:] x, real_t[:] y, real_t c, real_t s):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(real_t)
    cdef CBLAS_INT incy = y.strides[0] // sizeof(real_t)
    cdef CBLAS_INT N = x.shape[0]
    if N != y.shape[0]:
        raise ValueError("x and y must have the same length")

    if real_t is double:
        cblas_drot(N, &x[0], incx, &y[0], incy, c, s)
    elif real_t is float:
        cblas_srot(N, &x[0], incx, &y[0], incy, c, s)


def cblas_rotg(real_t a, real_t b):
    
        cdef real_t c, s, r
        if real_t is double:
            cblas_drotg(&a, &b, &c, &s)
        elif real_t is float:
            cblas_srotg(&a, &b, &c, &s)
        return c, s

def cblas_rotm(real_t[:] x, real_t[:] y, real_t[:] param):
    
        cdef CBLAS_INT incx = x.strides[0] // sizeof(real_t)
        cdef CBLAS_INT incy = y.strides[0] // sizeof(real_t)
        cdef CBLAS_INT N = x.shape[0]
        if N != y.shape[0]:
            raise ValueError("x and y must have the same length")
        if N != param.shape[0]:
            raise ValueError("x and param must have the same length")
    
        if real_t is double:
            cblas_drotm(N, &x[0], incx, &y[0], incy, &param[0])
        elif real_t is float:
            cblas_srotm(N, &x[0], incx, &y[0], incy, &param[0])


def cblas_rotmg(real_t d1, real_t d2, real_t x1, real_t y1):
    
        cdef real_t param[5]
        if real_t is double:
            cblas_drotmg(&d1, &d2, &x1, y1, &param[0])
        elif real_t is float:
            cblas_srotmg(&d1, &d2, &x1, y1, &param[0])
        return d1, d2, x1, param

def cblas_swap(scalar_t[:] x, scalar_t[:] y):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]
    if N != y.shape[0]:
        raise ValueError("x and y must have the same length")

    if scalar_t is double:
        cblas_dswap(N, &x[0], incx, &y[0], incy)
    elif scalar_t is float:
        cblas_sswap(N, &x[0], incx, &y[0], incy)
    elif scalar_t is dcomplex:
        cblas_zswap(N, <void*> &x[0], incx, <void*> &y[0], incy)
    elif scalar_t is fcomplex:
        cblas_cswap(N, <void*> &x[0], incx, <void*> &y[0], incy)

def cblas_scal(scalar_t[:] x, scalar_t factor):

    cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
    cdef CBLAS_INT N = x.shape[0]

    if scalar_t is double:
        cblas_dscal(N, factor, &x[0], incx)
    elif scalar_t is float:
        cblas_sscal(N, factor, &x[0], incx)
    elif scalar_t is dcomplex:
        cblas_zscal(N, <void*> &factor, <void*> &x[0], incx)
    elif scalar_t is fcomplex:
        cblas_cscal(N, <void*> &factor, <void*> &x[0], incx)
    


# Level 2

def cblas_gemv(scalar_t alpha, scalar_t[:,:] A, scalar_t[:] x, scalar_t beta, scalar_t[:] y=None):
    if A.strides[1] // sizeof(scalar_t) == 1:
        cblas_gemvC(alpha, A, x, beta, y)
    elif A.strides[0] // sizeof(scalar_t) == 1:
        cblas_gemvF(alpha, A, x, beta, y)
    else:
        raise ValueError("Matrix A must be contiguous in either C or F order")

# C-contig
def cblas_gemvC(scalar_t alpha, scalar_t[:,::1] A, scalar_t[:] x, scalar_t beta, scalar_t[:] y):
    
        cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT M = A.shape[0]
        cdef CBLAS_INT N = A.shape[1]
        cdef CBLAS_INT lda = A.strides[0] // sizeof(scalar_t)
    
        if scalar_t is double:
            cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, &A[0,0], lda, &x[0], incx, beta, &y[0], incy)
        elif scalar_t is float:
            cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, &A[0,0], lda, &x[0], incx, beta, &y[0], incy)
        elif scalar_t is dcomplex:
            cblas_zgemv(CblasRowMajor, CblasNoTrans, M, N, <void*> &alpha, <void*> &A[0,0], lda, <void*> &x[0], incx, <void*> &beta, <void*> &y[0], incy)
        elif scalar_t is fcomplex:
            cblas_cgemv(CblasRowMajor, CblasNoTrans, M, N, <void*> &alpha, <void*> &A[0,0], lda, <void*> &x[0], incx, <void*> &beta, <void*> &y[0], incy)

# F-contig
def cblas_gemvF(scalar_t alpha, scalar_t[::1,:] A, scalar_t[:] x, scalar_t beta, scalar_t[:] y):
    
        cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT M = A.shape[0]
        cdef CBLAS_INT N = A.shape[1]
        cdef CBLAS_INT lda = A.strides[1] // sizeof(scalar_t)
    
        if scalar_t is double:
            cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, alpha, &A[0,0], lda, &x[0], incx, beta, &y[0], incy)
        elif scalar_t is float:
            cblas_sgemv(CblasColMajor, CblasNoTrans, M, N, alpha, &A[0,0], lda, &x[0], incx, beta, &y[0], incy)
        elif scalar_t is dcomplex:
            cblas_zgemv(CblasColMajor, CblasNoTrans, M, N, <void*> &alpha, <void*> &A[0,0], lda, <void*> &x[0], incx, <void*> &beta, <void*> &y[0], incy)
        elif scalar_t is fcomplex:
            cblas_cgemv(CblasColMajor, CblasNoTrans, M, N, <void*> &alpha, <void*> &A[0,0], lda, <void*> &x[0], incx, <void*> &beta, <void*> &y[0], incy)


def cblas_ger(real_t alpha, real_t[:] x, real_t[:] y, real_t[:,:] A):
    cblas_gerc(alpha, x, y, A)
    
def cblas_gerc(scalar_t alpha, scalar_t[:] x, scalar_t[:] y, scalar_t[:,:] A):

    if A.strides[1] // sizeof(scalar_t) == 1:
        cblas_gercC(alpha, x, y, A)
    elif A.strides[0] // sizeof(scalar_t) == 1:
        cblas_gercF(alpha, x, y, A)
    else:
        raise ValueError("Matrix A must be contiguous in either C or F order")


def cblas_gercC(scalar_t alpha, scalar_t[:] x, scalar_t[:] y, scalar_t[:,::1] A):
    
        cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT M = A.shape[0]
        cdef CBLAS_INT N = A.shape[1]
        cdef CBLAS_INT lda = A.strides[0] // sizeof(scalar_t)
    
        if scalar_t is double:
            cblas_dger(CblasRowMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is float:
            cblas_sger(CblasRowMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is dcomplex:
            cblas_zgerc(CblasRowMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)
        elif scalar_t is fcomplex:
            cblas_cgerc(CblasRowMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)

def cblas_gercF(scalar_t alpha, scalar_t[:] x, scalar_t[:] y, scalar_t[::1,:] A):
    
        cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT M = A.shape[0]
        cdef CBLAS_INT N = A.shape[1]
        cdef CBLAS_INT lda = A.strides[1] // sizeof(scalar_t)
    
        if scalar_t is double:
            cblas_dger(CblasColMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is float:
            cblas_sger(CblasColMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is dcomplex:
            cblas_zgerc(CblasColMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)
        elif scalar_t is fcomplex:
            cblas_cgerc(CblasColMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)


def cblas_geru(scalar_t alpha, scalar_t[:] x, scalar_t[:] y, scalar_t[:,:] A):

    if A.strides[1] // sizeof(scalar_t) == 1:
        cblas_geruC(alpha, x, y, A)
    elif A.strides[0] // sizeof(scalar_t) == 1:
        cblas_geruF(alpha, x, y, A)
    else:
        raise ValueError("Matrix A must be contiguous in either C or F order")


def cblas_geruC(scalar_t alpha, scalar_t[:] x, scalar_t[:] y, scalar_t[:,::1] A):
    
        cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT M = A.shape[0]
        cdef CBLAS_INT N = A.shape[1]
        cdef CBLAS_INT lda = A.strides[0] // sizeof(scalar_t)
    
        if scalar_t is double:
            cblas_dger(CblasRowMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is float:
            cblas_sger(CblasRowMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is dcomplex:
            cblas_zgeru(CblasRowMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)
        elif scalar_t is fcomplex:
            cblas_cgeru(CblasRowMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)

def cblas_geruF(scalar_t alpha, scalar_t[:] x, scalar_t[:] y, scalar_t[::1,:] A):
    
        cdef CBLAS_INT incx = x.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT incy = y.strides[0] // sizeof(scalar_t)
        cdef CBLAS_INT M = A.shape[0]
        cdef CBLAS_INT N = A.shape[1]
        cdef CBLAS_INT lda = A.strides[1] // sizeof(scalar_t)
    
        if scalar_t is double:
            cblas_dger(CblasColMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is float:
            cblas_sger(CblasColMajor, M, N, alpha, &x[0], incx, &y[0], incy, &A[0,0], lda)
        elif scalar_t is dcomplex:
            cblas_zgeru(CblasColMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)
        elif scalar_t is fcomplex:
            cblas_cgeru(CblasColMajor, M, N, <void*> &alpha, <void*> &x[0], incx, <void*> &y[0], incy, <void*> &A[0,0], lda)


def cblas_gemm(scalar_t[:,:] A, scalar_t[:,:] B, scalar_t alpha=1.0, scalar_t beta=0.0, scalar_t[:,:] C=None, outcontig='C'):
    cdef CBLAS_ORDER la, lb, lc
    cdef CBLAS_INT M, N, K, lda, ldb, ldc
    cdef CBLAS_TRANSPOSE transa, transb
    cdef int return_array = 0
    
    if A.strides[1] // sizeof(scalar_t) == 1: # C contig
        la = CblasRowMajor
        lda = A.strides[0] // sizeof(scalar_t)
    elif A.strides[0] // sizeof(scalar_t) == 1: # F contig
        la = CblasColMajor
        lda = A.strides[1] // sizeof(scalar_t)
    else:
        raise ValueError("Matrix A must be contiguous in either C or F order")
    M = A.shape[0]
    K = A.shape[1]


    if B.strides[1] // sizeof(scalar_t) == 1: # C contig
        lb = CblasRowMajor
        ldb = B.strides[0] // sizeof(scalar_t)
    elif B.strides[0] // sizeof(scalar_t) == 1: # F contig
        lb = CblasColMajor
        ldb = B.strides[1] // sizeof(scalar_t)
    else:
        raise ValueError("Matrix B must be contiguous in either C or F order")
    
    N = B.shape[1]
    if K != B.shape[0]:
        raise ValueError("A and B must have the same inner dimension")
    
    if C is None:
        if beta == 0.0:
            npC = np.zeros_like(A, shape=(M, N), order=outcontig)
        else:
            npC = np.empty_like(A, shape=(M, N), order=outcontig)
        return_array = 1
        C = npC
        lc = CblasRowMajor if outcontig == 'C' else CblasColMajor
    else:
        if C.shape[0] != M or C.shape[1] != N:
            raise ValueError("C must have the same shape as the result")
        if C.strides[1] // sizeof(scalar_t) == 1: # C contig
            lc = CblasRowMajor
        elif C.strides[0] // sizeof(scalar_t) == 1: # F contig
            lc = CblasColMajor
        else:
            raise ValueError("Matrix C must be contiguous in either C or F order")

    if lc == CblasRowMajor:
        transa = CblasNoTrans if la == CblasRowMajor else CblasTrans
        transb = CblasNoTrans if lb == CblasRowMajor else CblasTrans
        ldc = C.strides[0] // sizeof(scalar_t)
    else:
        transa = CblasNoTrans if la == CblasColMajor else CblasTrans
        transb = CblasNoTrans if lb == CblasColMajor else CblasTrans
        ldc = C.strides[1] // sizeof(scalar_t)
    
    if scalar_t is double:
        cblas_dgemm(lc, transa, transb, M, N, K, alpha, &A[0,0], lda, &B[0,0], ldb, beta, &C[0,0], ldc)
    elif scalar_t is float:
        cblas_sgemm(lc, transa, transb, M, N, K, alpha, &A[0,0], lda, &B[0,0], ldb, beta, &C[0,0], ldc)
    elif scalar_t is dcomplex:
        cblas_zgemm(lc, transa, transb, M, N, K, <void*> &alpha, <void*> &A[0,0], lda, <void*> &B[0,0], ldb, <void*> &beta, <void*> &C[0,0], ldc)
    elif scalar_t is fcomplex:
        cblas_cgemm(lc, transa, transb, M, N, K, <void*> &alpha, <void*> &A[0,0], lda, <void*> &B[0,0], ldb, <void*> &beta, <void*> &C[0,0], ldc)
    
    if return_array:
        return npC