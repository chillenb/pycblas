cdef extern from "cblas.h" nogil:
    IF PYCBLAS_ILP64:
        ctypedef long long int CBLAS_INT
    ELSE:
        ctypedef int CBLAS_INT

    cpdef enum CBLAS_ORDER:
        CblasRowMajor
        CblasColMajor

    cpdef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans

    cpdef enum CBLAS_UPLO:
        CblasUpper
        CblasLower

    cpdef enum CBLAS_DIAG:
        CblasNonUnit
        CblasUnit

    cpdef enum CBLAS_SIDE:
        CblasLeft
        CblasRight

    float cblas_sdsdot(const CBLAS_INT N, const float alpha, const float* X, const CBLAS_INT incX, const float* Y, const CBLAS_INT incY)

    double cblas_dsdot(const CBLAS_INT N, const float* X, const CBLAS_INT incX, const float* Y, const CBLAS_INT incY)

    float cblas_sdot(const CBLAS_INT N, const float* X, const CBLAS_INT incX, const float* Y, const CBLAS_INT incY)

    double cblas_ddot(const CBLAS_INT N, const double* X, const CBLAS_INT incX, const double* Y, const CBLAS_INT incY)

    void cblas_cdotu_sub(const CBLAS_INT N, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* dotu)

    void cblas_cdotc_sub(const CBLAS_INT N, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* dotc)

    void cblas_zdotu_sub(const CBLAS_INT N, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* dotu)

    void cblas_zdotc_sub(const CBLAS_INT N, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* dotc)

    float cblas_snrm2(const CBLAS_INT N, const float* X, const CBLAS_INT incX)

    float cblas_sasum(const CBLAS_INT N, const float* X, const CBLAS_INT incX)

    double cblas_dnrm2(const CBLAS_INT N, const double* X, const CBLAS_INT incX)

    double cblas_dasum(const CBLAS_INT N, const double* X, const CBLAS_INT incX)

    float cblas_scnrm2(const CBLAS_INT N, const void* X, const CBLAS_INT incX)

    float cblas_scasum(const CBLAS_INT N, const void* X, const CBLAS_INT incX)

    double cblas_dznrm2(const CBLAS_INT N, const void* X, const CBLAS_INT incX)

    double cblas_dzasum(const CBLAS_INT N, const void* X, const CBLAS_INT incX)

    size_t cblas_isamax(const CBLAS_INT N, const float* X, const CBLAS_INT incX)

    size_t cblas_idamax(const CBLAS_INT N, const double* X, const CBLAS_INT incX)

    size_t cblas_icamax(const CBLAS_INT N, const void* X, const CBLAS_INT incX)

    size_t cblas_izamax(const CBLAS_INT N, const void* X, const CBLAS_INT incX)

    void cblas_sswap(const CBLAS_INT N, float* X, const CBLAS_INT incX, float* Y, const CBLAS_INT incY)

    void cblas_scopy(const CBLAS_INT N, const float* X, const CBLAS_INT incX, float* Y, const CBLAS_INT incY)

    void cblas_saxpy(const CBLAS_INT N, const float alpha, const float* X, const CBLAS_INT incX, float* Y, const CBLAS_INT incY)

    void cblas_dswap(const CBLAS_INT N, double* X, const CBLAS_INT incX, double* Y, const CBLAS_INT incY)

    void cblas_dcopy(const CBLAS_INT N, const double* X, const CBLAS_INT incX, double* Y, const CBLAS_INT incY)

    void cblas_daxpy(const CBLAS_INT N, const double alpha, const double* X, const CBLAS_INT incX, double* Y, const CBLAS_INT incY)

    void cblas_cswap(const CBLAS_INT N, void* X, const CBLAS_INT incX, void* Y, const CBLAS_INT incY)

    void cblas_ccopy(const CBLAS_INT N, const void* X, const CBLAS_INT incX, void* Y, const CBLAS_INT incY)

    void cblas_caxpy(const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, void* Y, const CBLAS_INT incY)

    void cblas_zswap(const CBLAS_INT N, void* X, const CBLAS_INT incX, void* Y, const CBLAS_INT incY)

    void cblas_zcopy(const CBLAS_INT N, const void* X, const CBLAS_INT incX, void* Y, const CBLAS_INT incY)

    void cblas_zaxpy(const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, void* Y, const CBLAS_INT incY)

    void cblas_srotg(float* a, float* b, float* c, float* s)

    void cblas_srotmg(float* d1, float* d2, float* b1, const float b2, float* P)

    void cblas_srot(const CBLAS_INT N, float* X, const CBLAS_INT incX, float* Y, const CBLAS_INT incY, const float c, const float s)

    void cblas_srotm(const CBLAS_INT N, float* X, const CBLAS_INT incX, float* Y, const CBLAS_INT incY, const float* P)

    void cblas_drotg(double* a, double* b, double* c, double* s)

    void cblas_drotmg(double* d1, double* d2, double* b1, const double b2, double* P)

    void cblas_drot(const CBLAS_INT N, double* X, const CBLAS_INT incX, double* Y, const CBLAS_INT incY, const double c, const double s)

    void cblas_drotm(const CBLAS_INT N, double* X, const CBLAS_INT incX, double* Y, const CBLAS_INT incY, const double* P)

    void cblas_sscal(const CBLAS_INT N, const float alpha, float* X, const CBLAS_INT incX)

    void cblas_dscal(const CBLAS_INT N, const double alpha, double* X, const CBLAS_INT incX)

    void cblas_cscal(const CBLAS_INT N, const void* alpha, void* X, const CBLAS_INT incX)

    void cblas_zscal(const CBLAS_INT N, const void* alpha, void* X, const CBLAS_INT incX)

    void cblas_csscal(const CBLAS_INT N, const float alpha, void* X, const CBLAS_INT incX)

    void cblas_zdscal(const CBLAS_INT N, const double alpha, void* X, const CBLAS_INT incX)

    void cblas_sgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const float alpha, const float* A, const CBLAS_INT lda, const float* X, const CBLAS_INT incX, const float beta, float* Y, const CBLAS_INT incY)

    void cblas_sgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU, const float alpha, const float* A, const CBLAS_INT lda, const float* X, const CBLAS_INT incX, const float beta, float* Y, const CBLAS_INT incY)

    void cblas_strmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const float* A, const CBLAS_INT lda, float* X, const CBLAS_INT incX)

    void cblas_stbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const float* A, const CBLAS_INT lda, float* X, const CBLAS_INT incX)

    void cblas_stpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const float* Ap, float* X, const CBLAS_INT incX)

    void cblas_strsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const float* A, const CBLAS_INT lda, float* X, const CBLAS_INT incX)

    void cblas_stbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const float* A, const CBLAS_INT lda, float* X, const CBLAS_INT incX)

    void cblas_stpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const float* Ap, float* X, const CBLAS_INT incX)

    void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const double alpha, const double* A, const CBLAS_INT lda, const double* X, const CBLAS_INT incX, const double beta, double* Y, const CBLAS_INT incY)

    void cblas_dgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU, const double alpha, const double* A, const CBLAS_INT lda, const double* X, const CBLAS_INT incX, const double beta, double* Y, const CBLAS_INT incY)

    void cblas_dtrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const double* A, const CBLAS_INT lda, double* X, const CBLAS_INT incX)

    void cblas_dtbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const double* A, const CBLAS_INT lda, double* X, const CBLAS_INT incX)

    void cblas_dtpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const double* Ap, double* X, const CBLAS_INT incX)

    void cblas_dtrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const double* A, const CBLAS_INT lda, double* X, const CBLAS_INT incX)

    void cblas_dtbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const double* A, const CBLAS_INT lda, double* X, const CBLAS_INT incX)

    void cblas_dtpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const double* Ap, double* X, const CBLAS_INT incX)

    void cblas_cgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_cgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_ctrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ctbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ctpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* Ap, void* X, const CBLAS_INT incX)

    void cblas_ctrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ctbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ctpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* Ap, void* X, const CBLAS_INT incX)

    void cblas_zgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_zgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_ztrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ztbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ztpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* Ap, void* X, const CBLAS_INT incX)

    void cblas_ztrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ztbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K, const void* A, const CBLAS_INT lda, void* X, const CBLAS_INT incX)

    void cblas_ztpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT N, const void* Ap, void* X, const CBLAS_INT incX)

    void cblas_ssymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const float* A, const CBLAS_INT lda, const float* X, const CBLAS_INT incX, const float beta, float* Y, const CBLAS_INT incY)

    void cblas_ssbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT K, const float alpha, const float* A, const CBLAS_INT lda, const float* X, const CBLAS_INT incX, const float beta, float* Y, const CBLAS_INT incY)

    void cblas_sspmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const float* Ap, const float* X, const CBLAS_INT incX, const float beta, float* Y, const CBLAS_INT incY)

    void cblas_sger(const CBLAS_ORDER order, const CBLAS_INT M, const CBLAS_INT N, const float alpha, const float* X, const CBLAS_INT incX, const float* Y, const CBLAS_INT incY, float* A, const CBLAS_INT lda)

    void cblas_ssyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const float* X, const CBLAS_INT incX, float* A, const CBLAS_INT lda)

    void cblas_sspr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const float* X, const CBLAS_INT incX, float* Ap)

    void cblas_ssyr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const float* X, const CBLAS_INT incX, const float* Y, const CBLAS_INT incY, float* A, const CBLAS_INT lda)

    void cblas_sspr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const float* X, const CBLAS_INT incX, const float* Y, const CBLAS_INT incY, float* A)

    void cblas_dsymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const double* A, const CBLAS_INT lda, const double* X, const CBLAS_INT incX, const double beta, double* Y, const CBLAS_INT incY)

    void cblas_dsbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT K, const double alpha, const double* A, const CBLAS_INT lda, const double* X, const CBLAS_INT incX, const double beta, double* Y, const CBLAS_INT incY)

    void cblas_dspmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const double* Ap, const double* X, const CBLAS_INT incX, const double beta, double* Y, const CBLAS_INT incY)

    void cblas_dger(const CBLAS_ORDER order, const CBLAS_INT M, const CBLAS_INT N, const double alpha, const double* X, const CBLAS_INT incX, const double* Y, const CBLAS_INT incY, double* A, const CBLAS_INT lda)

    void cblas_dsyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const double* X, const CBLAS_INT incX, double* A, const CBLAS_INT lda)

    void cblas_dspr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const double* X, const CBLAS_INT incX, double* Ap)

    void cblas_dsyr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const double* X, const CBLAS_INT incX, const double* Y, const CBLAS_INT incY, double* A, const CBLAS_INT lda)

    void cblas_dspr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const double* X, const CBLAS_INT incX, const double* Y, const CBLAS_INT incY, double* A)

    void cblas_chemv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_chbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_chpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* Ap, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_cgeru(const CBLAS_ORDER order, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* A, const CBLAS_INT lda)

    void cblas_cgerc(const CBLAS_ORDER order, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* A, const CBLAS_INT lda)

    void cblas_cher(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const void* X, const CBLAS_INT incX, void* A, const CBLAS_INT lda)

    void cblas_chpr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const float alpha, const void* X, const CBLAS_INT incX, void* A)

    void cblas_cher2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* A, const CBLAS_INT lda)

    void cblas_chpr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* Ap)

    void cblas_zhemv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_zhbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_zhpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* Ap, const void* X, const CBLAS_INT incX, const void* beta, void* Y, const CBLAS_INT incY)

    void cblas_zgeru(const CBLAS_ORDER order, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* A, const CBLAS_INT lda)

    void cblas_zgerc(const CBLAS_ORDER order, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* A, const CBLAS_INT lda)

    void cblas_zher(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const void* X, const CBLAS_INT incX, void* A, const CBLAS_INT lda)

    void cblas_zhpr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const double alpha, const void* X, const CBLAS_INT incX, void* A)

    void cblas_zher2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* A, const CBLAS_INT lda)

    void cblas_zhpr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_INT N, const void* alpha, const void* X, const CBLAS_INT incX, const void* Y, const CBLAS_INT incY, void* Ap)

    void cblas_sgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT K, const float alpha, const float* A, const CBLAS_INT lda, const float* B, const CBLAS_INT ldb, const float beta, float* C, const CBLAS_INT ldc)

    void cblas_ssymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N, const float alpha, const float* A, const CBLAS_INT lda, const float* B, const CBLAS_INT ldb, const float beta, float* C, const CBLAS_INT ldc)

    void cblas_ssyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const float alpha, const float* A, const CBLAS_INT lda, const float beta, float* C, const CBLAS_INT ldc)

    void cblas_ssyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const float alpha, const float* A, const CBLAS_INT lda, const float* B, const CBLAS_INT ldb, const float beta, float* C, const CBLAS_INT ldc)

    void cblas_strmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const float alpha, const float* A, const CBLAS_INT lda, float* B, const CBLAS_INT ldb)

    void cblas_strsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const float alpha, const float* A, const CBLAS_INT lda, float* B, const CBLAS_INT ldb)

    void cblas_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT K, const double alpha, const double* A, const CBLAS_INT lda, const double* B, const CBLAS_INT ldb, const double beta, double* C, const CBLAS_INT ldc)

    void cblas_dsymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N, const double alpha, const double* A, const CBLAS_INT lda, const double* B, const CBLAS_INT ldb, const double beta, double* C, const CBLAS_INT ldc)

    void cblas_dsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const double alpha, const double* A, const CBLAS_INT lda, const double beta, double* C, const CBLAS_INT ldc)

    void cblas_dsyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const double alpha, const double* A, const CBLAS_INT lda, const double* B, const CBLAS_INT ldb, const double beta, double* C, const CBLAS_INT ldc)

    void cblas_dtrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const double alpha, const double* A, const CBLAS_INT lda, double* B, const CBLAS_INT ldb)

    void cblas_dtrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const double alpha, const double* A, const CBLAS_INT lda, double* B, const CBLAS_INT ldb)

    void cblas_cgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_csymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_csyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_csyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_ctrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, void* B, const CBLAS_INT ldb)

    void cblas_ctrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, void* B, const CBLAS_INT ldb)

    void cblas_zgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_zsymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_zsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_zsyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_ztrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, void* B, const CBLAS_INT ldb)

    void cblas_ztrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, void* B, const CBLAS_INT ldb)

    void cblas_chemm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_cherk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const float alpha, const void* A, const CBLAS_INT lda, const float beta, void* C, const CBLAS_INT ldc)

    void cblas_cher2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const float beta, void* C, const CBLAS_INT ldc)

    void cblas_zhemm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const void* beta, void* C, const CBLAS_INT ldc)

    void cblas_zherk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const double alpha, const void* A, const CBLAS_INT lda, const double beta, void* C, const CBLAS_INT ldc)

    void cblas_zher2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K, const void* alpha, const void* A, const CBLAS_INT lda, const void* B, const CBLAS_INT ldb, const double beta, void* C, const CBLAS_INT ldc)
