# cython: language_level=3

from cblas cimport *

ctypedef double complex dcomplex
ctypedef float complex fcomplex

ctypedef fused scalar_t:
    double
    float
    dcomplex
    fcomplex


def cblas_scal(scalar_t[:] arr, scalar_t factor):

    cdef int type_stride = arr.strides[0] // sizeof(scalar_t)
    cdef int N = arr.shape[0]

    if scalar_t is double:
        cblas_dscal(N, factor, &arr[0], type_stride)
    elif scalar_t is float:
        cblas_sscal(N, factor, &arr[0], type_stride)
    elif scalar_t is dcomplex:
        cblas_zscal(N, <void*> &factor, <void*> &arr[0], type_stride)
    elif scalar_t is fcomplex:
        cblas_cscal(N, <void*> &factor, <void*> &arr[0], type_stride)
    



    
