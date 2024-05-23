# cython: language_level=3

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
    



    
