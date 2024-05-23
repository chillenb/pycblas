#include "cblas.h"

#include <complex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

#define IS_C_ORDER(x) ((x).stride(1) == 1)
#define IS_F_ORDER(x) ((x).stride(0) == 1)

#define CHK_DIMS_L2(M, N, X, Y)                                                \
  if ((Y).shape(0) != M || (X).shape(0) != N) {                                \
    throw std::length_error(                                                   \
        "Incompatible shapes: (" + std::to_string(M) + ", " +                  \
        std::to_string(N) + "), (" + std::to_string((X).shape(0)) + ") -> (" + \
        std::to_string((Y).shape(0)) + ") in " + __func__);                    \
  }

#define CHK_DIMS_SYM_L2(M, N, X, Y)                                            \
  if ((M) != (N) || (Y).shape(0) != (N) || (X).shape(0) != (N)) {              \
    throw std::length_error(                                                   \
        "Incompatible shapes: (" + std::to_string(M) + ", " +                  \
        std::to_string(N) + "), (" + std::to_string((X).shape(0)) + ") -> (" + \
        std::to_string((Y).shape(0)) + ") in " + __func__);                    \
  }

#define CHK_DIMS_L3(M, N, K, A, B, C)                                          \
  if ((A).shape(0) != M || (A).shape(1) != K || (B).shape(0) != K ||           \
      (B).shape(1) != N || (C).shape(0) != M || (C).shape(1) != N) {           \
    throw std::length_error("Incompatible shapes: (" +                 \
                            std::to_string((A).shape(0)) + ", " +               \
                            std::to_string((A).shape(1)) + "), (" +             \
                            std::to_string((B).shape(0)) + ", " +               \
                            std::to_string((B).shape(1)) + ") -> (" +             \
                            std::to_string((C).shape(0)) + ", " +               \
                            std::to_string((C).shape(1)) + ") in " + __func__); \
  }

#define ORDER_A                                                                \
  if (IS_C_ORDER(A)) {                                                         \
    lda = A.stride(0);                                                         \
    order = CblasRowMajor;                                                     \
  } else if (IS_F_ORDER(A)) {                                                  \
    lda = A.stride(1);                                                         \
    order = CblasColMajor;                                                     \
  } else {                                                                     \
    throw std::runtime_error("A must be C or F contiguous");                   \
  }

#define ORDER_A3                                                               \
  if (IS_C_ORDER(A)) {                                                         \
    lda = A.stride(0);                                                         \
    transa = (order == CblasRowMajor) ? CblasNoTrans : CblasTrans;             \
  } else if (IS_F_ORDER(A)) {                                                  \
    lda = A.stride(1);                                                         \
    transa = (order == CblasColMajor) ? CblasNoTrans : CblasTrans;             \
  } else {                                                                     \
    throw std::runtime_error("A must be C or F contiguous");                   \
  }

#define ORDER_B3                                                               \
  if (IS_C_ORDER(B)) {                                                         \
    ldb = B.stride(0);                                                         \
    transb = (order == CblasRowMajor) ? CblasNoTrans : CblasTrans;             \
  } else if (IS_F_ORDER(B)) {                                                  \
    ldb = B.stride(1);                                                         \
    transb = (order == CblasColMajor) ? CblasNoTrans : CblasTrans;             \
  } else {                                                                     \
    throw std::runtime_error("B must be C or F contiguous");                   \
  }

#define ORDER_C3                                                               \
  if (IS_C_ORDER(C)) {                                                         \
    ldc = C.stride(0);                                                         \
    order = CblasRowMajor;                                                     \
  } else if (IS_F_ORDER(C)) {                                                  \
    ldc = C.stride(1);                                                         \
    order = CblasColMajor;                                                     \
  } else {                                                                     \
    throw std::runtime_error("C must be C or F contiguous");                   \
  }

CBLAS_UPLO _cb_uplo(char uplo) {
  switch (uplo) {
  case 'U':
    return CblasUpper;
  case 'L':
    return CblasLower;
  default:
    throw std::invalid_argument("Invalid uplo");
  }
}

float _cb_sdot(nb::ndarray<float, nb::ndim<1>> X,
               nb::ndarray<float, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  return cblas_sdot(N, X.data(), incX, Y.data(), incY);
}

float _cb_ddot(nb::ndarray<double, nb::ndim<1>> X,
               nb::ndarray<double, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  return cblas_ddot(N, X.data(), incX, Y.data(), incY);
}

std::complex<float>
_cb_cdotu_sub(nb::ndarray<std::complex<float>, nb::ndim<1>> X,
              nb::ndarray<std::complex<float>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  std::complex<float> res;
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_cdotu_sub(N, reinterpret_cast<const void *>(X.data()), incX,
                  reinterpret_cast<const void *>(Y.data()), incY,
                  reinterpret_cast<void *>(&res));
  return res;
}

std::complex<float>
_cb_cdotc_sub(nb::ndarray<std::complex<float>, nb::ndim<1>> X,
              nb::ndarray<std::complex<float>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  std::complex<float> res;
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_cdotc_sub(N, reinterpret_cast<const void *>(X.data()), incX,
                  reinterpret_cast<const void *>(Y.data()), incY,
                  reinterpret_cast<void *>(&res));
  return res;
}

std::complex<double>
_cb_zdotu_sub(nb::ndarray<std::complex<double>, nb::ndim<1>> X,
              nb::ndarray<std::complex<double>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  std::complex<double> res;
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_cdotu_sub(N, reinterpret_cast<const void *>(X.data()), incX,
                  reinterpret_cast<const void *>(Y.data()), incY,
                  reinterpret_cast<void *>(&res));
  return res;
}

std::complex<double>
_cb_zdotc_sub(nb::ndarray<std::complex<double>, nb::ndim<1>> X,
              nb::ndarray<std::complex<double>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  std::complex<double> res;
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_cdotc_sub(N, reinterpret_cast<const void *>(X.data()), incX,
                  reinterpret_cast<const void *>(Y.data()), incY,
                  reinterpret_cast<void *>(&res));
  return res;
}

float _cb_snrm2(nb::ndarray<float, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_snrm2(N, X.data(), incX);
}

double _cb_dnrm2(nb::ndarray<double, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_dnrm2(N, X.data(), incX);
}

float _cb_scnrm2(nb::ndarray<std::complex<float>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_scnrm2(N, reinterpret_cast<const void *>(X.data()), incX);
}

double _cb_dznrm2(nb::ndarray<std::complex<double>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_dznrm2(N, reinterpret_cast<const void *>(X.data()), incX);
}

float _cb_sasum(nb::ndarray<float, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_sasum(N, X.data(), incX);
}

double _cb_dasum(nb::ndarray<double, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_dasum(N, X.data(), incX);
}

float _cb_scasum(nb::ndarray<std::complex<float>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_scasum(N, reinterpret_cast<const void *>(X.data()), incX);
}

double _cb_dzasum(nb::ndarray<std::complex<double>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_dzasum(N, reinterpret_cast<const void *>(X.data()), incX);
}

size_t _cb_isamax(nb::ndarray<float, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_isamax(N, X.data(), incX);
}

size_t _cb_idamax(nb::ndarray<double, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_idamax(N, X.data(), incX);
}

size_t _cb_icamax(nb::ndarray<std::complex<float>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_icamax(N, reinterpret_cast<const void *>(X.data()), incX);
}

size_t _cb_izamax(nb::ndarray<std::complex<double>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  return cblas_izamax(N, reinterpret_cast<const void *>(X.data()), incX);
}

void _cb_sswap(nb::ndarray<float, nb::ndim<1>> X,
               nb::ndarray<float, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_sswap(N, X.data(), incX, Y.data(), incY);
}

void _cb_dswap(nb::ndarray<double, nb::ndim<1>> X,
               nb::ndarray<double, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_dswap(N, X.data(), incX, Y.data(), incY);
}

void _cb_cswap(nb::ndarray<std::complex<float>, nb::ndim<1>> X,
               nb::ndarray<std::complex<float>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_cswap(N, reinterpret_cast<void *>(X.data()), incX,
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_zswap(nb::ndarray<std::complex<double>, nb::ndim<1>> X,
               nb::ndarray<std::complex<double>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_zswap(N, reinterpret_cast<void *>(X.data()), incX,
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_scopy(nb::ndarray<float, nb::ndim<1>> X,
               nb::ndarray<float, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_scopy(N, X.data(), incX, Y.data(), incY);
}

void _cb_dcopy(nb::ndarray<double, nb::ndim<1>> X,
               nb::ndarray<double, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_dcopy(N, X.data(), incX, Y.data(), incY);
}

void _cb_ccopy(nb::ndarray<std::complex<float>, nb::ndim<1>> X,
               nb::ndarray<std::complex<float>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_ccopy(N, reinterpret_cast<void *>(X.data()), incX,
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_zcopy(nb::ndarray<std::complex<double>, nb::ndim<1>> X,
               nb::ndarray<std::complex<double>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_zcopy(N, reinterpret_cast<void *>(X.data()), incX,
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_saxpy(float alpha, nb::ndarray<float, nb::ndim<1>> X,
               nb::ndarray<float, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_saxpy(N, alpha, X.data(), incX, Y.data(), incY);
}

void _cb_daxpy(double alpha, nb::ndarray<double, nb::ndim<1>> X,
               nb::ndarray<double, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_daxpy(N, alpha, X.data(), incX, Y.data(), incY);
}

void _cb_caxpy(std::complex<float> alpha,
               nb::ndarray<std::complex<float>, nb::ndim<1>> X,
               nb::ndarray<std::complex<float>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_caxpy(N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(X.data()), incX,
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_zaxpy(std::complex<double> alpha,
               nb::ndarray<std::complex<double>, nb::ndim<1>> X,
               nb::ndarray<std::complex<double>, nb::ndim<1>> Y) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_zaxpy(N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(X.data()), incX,
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_srot(nb::ndarray<float, nb::ndim<1>> X,
              nb::ndarray<float, nb::ndim<1>> Y, float c, float s) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_srot(N, X.data(), incX, Y.data(), incY, c, s);
}

void _cb_drot(nb::ndarray<double, nb::ndim<1>> X,
              nb::ndarray<double, nb::ndim<1>> Y, double c, double s) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  if (Y.shape(0) != N) {
    throw std::length_error("Incompatible shapes");
  }
  cblas_drot(N, X.data(), incX, Y.data(), incY, c, s);
}

void _cb_sscal(float alpha, nb::ndarray<float, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  cblas_sscal(N, alpha, X.data(), incX);
}

void _cb_dscal(double alpha, nb::ndarray<double, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  cblas_dscal(N, alpha, X.data(), incX);
}

void _cb_cscal(std::complex<float> alpha,
               nb::ndarray<std::complex<float>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  cblas_cscal(N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<void *>(X.data()), incX);
}

void _cb_zscal(std::complex<double> alpha,
               nb::ndarray<std::complex<double>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  cblas_zscal(N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<void *>(X.data()), incX);
}

void _cb_csscal(float alpha, nb::ndarray<std::complex<float>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  cblas_csscal(N, alpha, reinterpret_cast<void *>(X.data()), incX);
}

void _cb_zdscal(double alpha,
                nb::ndarray<std::complex<double>, nb::ndim<1>> X) {
  CBLAS_INT N = X.shape(0);
  CBLAS_INT incX = X.stride(0);
  cblas_zdscal(N, alpha, reinterpret_cast<void *>(X.data()), incX);
}

void _cb_sgemv(nb::ndarray<float, nb::ndim<2>> A,
               nb::ndarray<float, nb::ndim<1>> X,
               nb::ndarray<float, nb::ndim<1>> Y, float alpha, float beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_sgemv(order, CblasNoTrans, M, N, alpha, A.data(), lda, X.data(), incX,
              beta, Y.data(), incY);
}

void _cb_dgemv(nb::ndarray<double, nb::ndim<2>> A,
               nb::ndarray<double, nb::ndim<1>> X,
               nb::ndarray<double, nb::ndim<1>> Y, double alpha, double beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_dgemv(order, CblasNoTrans, M, N, alpha, A.data(), lda, X.data(), incX,
              beta, Y.data(), incY);
}

void _cb_cgemv(nb::ndarray<std::complex<float>, nb::ndim<2>> A,
               nb::ndarray<std::complex<float>, nb::ndim<1>> X,
               nb::ndarray<std::complex<float>, nb::ndim<1>> Y,
               std::complex<float> alpha, std::complex<float> beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_cgemv(order, CblasNoTrans, M, N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(A.data()), lda,
              reinterpret_cast<const void *>(X.data()), incX,
              reinterpret_cast<const void *>(&beta),
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_zgemv(nb::ndarray<std::complex<double>, nb::ndim<2>> A,
               nb::ndarray<std::complex<double>, nb::ndim<1>> X,
               nb::ndarray<std::complex<double>, nb::ndim<1>> Y,
               std::complex<double> alpha, std::complex<double> beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_zgemv(order, CblasNoTrans, M, N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(A.data()), lda,
              reinterpret_cast<const void *>(X.data()), incX,
              reinterpret_cast<const void *>(&beta),
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_chemv(char uplo, nb::ndarray<std::complex<float>, nb::ndim<2>> A,
               nb::ndarray<std::complex<float>, nb::ndim<1>> X,
               nb::ndarray<std::complex<float>, nb::ndim<1>> Y,
               std::complex<float> alpha, std::complex<float> beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_SYM_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_chemv(order, _cb_uplo(uplo), N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(A.data()), lda,
              reinterpret_cast<const void *>(X.data()), incX,
              reinterpret_cast<const void *>(&beta),
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_zhemv(char uplo, nb::ndarray<std::complex<double>, nb::ndim<2>> A,
               nb::ndarray<std::complex<double>, nb::ndim<1>> X,
               nb::ndarray<std::complex<double>, nb::ndim<1>> Y,
               std::complex<double> alpha, std::complex<double> beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_SYM_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_zhemv(order, _cb_uplo(uplo), N, reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(A.data()), lda,
              reinterpret_cast<const void *>(X.data()), incX,
              reinterpret_cast<const void *>(&beta),
              reinterpret_cast<void *>(Y.data()), incY);
}

void _cb_ssymv(char uplo, nb::ndarray<float, nb::ndim<2>> A,
               nb::ndarray<float, nb::ndim<2>> X,
               nb::ndarray<float, nb::ndim<2>> Y, float alpha, float beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_SYM_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_ssymv(order, _cb_uplo(uplo), N, alpha, A.data(), lda, X.data(), incX,
              beta, Y.data(), incY);
}

void _cb_dsymv(char uplo, nb::ndarray<double, nb::ndim<2>> A,
               nb::ndarray<double, nb::ndim<2>> X,
               nb::ndarray<double, nb::ndim<2>> Y, double alpha, double beta) {
  CBLAS_INT M = A.shape(0);
  CBLAS_INT N = A.shape(1);
  CBLAS_INT lda;
  CBLAS_ORDER order;
  ORDER_A;
  CHK_DIMS_SYM_L2(M, N, X, Y);
  CBLAS_INT incX = X.stride(0);
  CBLAS_INT incY = Y.stride(0);
  cblas_dsymv(order, _cb_uplo(uplo), N, alpha, A.data(), lda, X.data(), incX,
              beta, Y.data(), incY);
}

void _cb_sgemm(nb::ndarray<float, nb::ndim<2>> A,
               nb::ndarray<float, nb::ndim<2>> B,
               nb::ndarray<float, nb::ndim<2>> C, float alpha, float beta) {
  CBLAS_INT M = C.shape(0);
  CBLAS_INT N = C.shape(1);
  CBLAS_INT K = A.shape(1);
  CBLAS_INT lda, ldb, ldc;
  CBLAS_TRANSPOSE transa, transb;
  CBLAS_ORDER order;
  ORDER_C3;
  ORDER_A3;
  ORDER_B3;
  CHK_DIMS_L3(M, N, K, A, B, C);
  cblas_sgemm(order, transa, transb, M, N, K, alpha, A.data(), lda, B.data(),
              ldb, beta, C.data(), ldc);
}

void _cb_dgemm(nb::ndarray<double, nb::ndim<2>> A,
               nb::ndarray<double, nb::ndim<2>> B,
               nb::ndarray<double, nb::ndim<2>> C, double alpha, double beta) {
  CBLAS_INT M = C.shape(0);
  CBLAS_INT N = C.shape(1);
  CBLAS_INT K = A.shape(1);
  CBLAS_INT lda, ldb, ldc;
  CBLAS_TRANSPOSE transa, transb;
  CBLAS_ORDER order;
  ORDER_C3;
  ORDER_A3;
  ORDER_B3;
  CHK_DIMS_L3(M, N, K, A, B, C);
  cblas_dgemm(order, transa, transb, M, N, K, alpha, A.data(), lda, B.data(),
              ldb, beta, C.data(), ldc);
}

void _cb_cgemm(nb::ndarray<std::complex<float>, nb::ndim<2>> A,
               nb::ndarray<std::complex<float>, nb::ndim<2>> B,
               nb::ndarray<std::complex<float>, nb::ndim<2>> C,
               std::complex<float> alpha, std::complex<float> beta) {
  CBLAS_INT M = C.shape(0);
  CBLAS_INT N = C.shape(1);
  CBLAS_INT K = A.shape(1);
  CBLAS_INT lda, ldb, ldc;
  CBLAS_TRANSPOSE transa, transb;
  CBLAS_ORDER order;
  ORDER_C3;
  ORDER_A3;
  ORDER_B3;
  CHK_DIMS_L3(M, N, K, A, B, C);
  cblas_cgemm(order, transa, transb, M, N, K,
              reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(A.data()), lda,
              reinterpret_cast<const void *>(B.data()), ldb,
              reinterpret_cast<const void *>(&beta),
              reinterpret_cast<void *>(C.data()), ldc);
}

void _cb_zgemm(nb::ndarray<std::complex<double>, nb::ndim<2>> A,
               nb::ndarray<std::complex<double>, nb::ndim<2>> B,
               nb::ndarray<std::complex<double>, nb::ndim<2>> C,
               std::complex<double> alpha, std::complex<double> beta) {
  CBLAS_INT M = C.shape(0);
  CBLAS_INT N = C.shape(1);
  CBLAS_INT K = A.shape(1);
  CBLAS_INT lda, ldb, ldc;
  CBLAS_TRANSPOSE transa, transb;
  CBLAS_ORDER order;
  ORDER_C3;
  ORDER_A3;
  ORDER_B3;
  CHK_DIMS_L3(M, N, K, A, B, C);
  cblas_zgemm(order, transa, transb, M, N, K,
              reinterpret_cast<const void *>(&alpha),
              reinterpret_cast<const void *>(A.data()), lda,
              reinterpret_cast<const void *>(B.data()), ldb,
              reinterpret_cast<const void *>(&beta),
              reinterpret_cast<void *>(C.data()), ldc);
}

NB_MODULE(pycblas, m) {
  m.def("sdot", &_cb_sdot, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("ddot", &_cb_ddot, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("cdotu", &_cb_cdotu_sub, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("cdotc", &_cb_cdotc_sub, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("zdotu", &_cb_zdotu_sub, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("zdotc", &_cb_zdotc_sub, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("snrm2", &_cb_snrm2, "X"_a.noconvert());
  m.def("dnrm2", &_cb_dnrm2, "X"_a.noconvert());
  m.def("scnrm2", &_cb_scnrm2, "X"_a.noconvert());
  m.def("dznrm2", &_cb_dznrm2, "X"_a.noconvert());
  m.def("sasum", &_cb_sasum, "X"_a.noconvert());
  m.def("dasum", &_cb_dasum, "X"_a.noconvert());
  m.def("scasum", &_cb_scasum, "X"_a.noconvert());
  m.def("dzasum", &_cb_dzasum, "X"_a.noconvert());
  m.def("isamax", &_cb_isamax, "X"_a.noconvert());
  m.def("idamax", &_cb_idamax, "X"_a.noconvert());
  m.def("icamax", &_cb_icamax, "X"_a.noconvert());
  m.def("izamax", &_cb_izamax, "X"_a.noconvert());
  m.def("sswap", &_cb_sswap, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("dswap", &_cb_dswap, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("cswap", &_cb_cswap, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("zswap", &_cb_zswap, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("scopy", &_cb_scopy, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("dcopy", &_cb_dcopy, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("ccopy", &_cb_ccopy, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("zcopy", &_cb_zcopy, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("saxpy", &_cb_saxpy, "alpha"_a, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("daxpy", &_cb_daxpy, "alpha"_a, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("caxpy", &_cb_caxpy, "alpha"_a, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("zaxpy", &_cb_zaxpy, "alpha"_a, "X"_a.noconvert(), "Y"_a.noconvert());
  m.def("srot", &_cb_srot, "X"_a.noconvert(), "Y"_a.noconvert(), "c"_a, "s"_a);
  m.def("drot", &_cb_drot, "X"_a.noconvert(), "Y"_a.noconvert(), "c"_a, "s"_a);
  m.def("sscal", &_cb_sscal, "alpha"_a, "X"_a.noconvert());
  m.def("dscal", &_cb_dscal, "alpha"_a, "X"_a.noconvert());
  m.def("cscal", &_cb_cscal, "alpha"_a, "X"_a.noconvert());
  m.def("zscal", &_cb_zscal, "alpha"_a, "X"_a.noconvert());
  m.def("csscal", &_cb_csscal, "alpha"_a, "X"_a.noconvert());
  m.def("zdscal", &_cb_zdscal, "alpha"_a, "X"_a.noconvert());

  m.def("sgemv", &_cb_sgemv, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = 1.0f, "beta"_a = 0.0f);
  m.def("dgemv", &_cb_dgemv, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = 1.0, "beta"_a = 0.0);
  m.def("cgemv", &_cb_cgemv, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = std::complex<float>(1.0f),
        "beta"_a = std::complex<float>(0.0f));
  m.def("zgemv", &_cb_zgemv, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = std::complex<double>(1.0),
        "beta"_a = std::complex<double>(0.0));

  m.def("ssymv", &_cb_ssymv, "uplo"_a, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = 1.0f, "beta"_a = 0.0f);
  m.def("dsymv", &_cb_dsymv, "uplo"_a, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = 1.0, "beta"_a = 0.0);
  m.def("chemv", &_cb_chemv, "uplo"_a, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = std::complex<float>(1.0f),
        "beta"_a = std::complex<float>(0.0f));
  m.def("zhemv", &_cb_zhemv, "uplo"_a, "A"_a.noconvert(), "X"_a.noconvert(),
        "Y"_a.noconvert(), "alpha"_a = std::complex<double>(1.0),
        "beta"_a = std::complex<double>(0.0));

  m.def("sgemm", &_cb_sgemm, "A"_a.noconvert(), "B"_a.noconvert(),
        "C"_a.noconvert(), "alpha"_a = 1.0f, "beta"_a = 0.0f);
  m.def("dgemm", &_cb_dgemm, "A"_a.noconvert(), "B"_a.noconvert(),
        "C"_a.noconvert(), "alpha"_a = 1.0, "beta"_a = 0.0);
  m.def("cgemm", &_cb_cgemm, "A"_a.noconvert(), "B"_a.noconvert(),
        "C"_a.noconvert(), "alpha"_a = std::complex<float>(1.0f),
        "beta"_a = std::complex<float>(0.0f));
  m.def("zgemm", &_cb_zgemm, "A"_a.noconvert(), "B"_a.noconvert(),
        "C"_a.noconvert(), "alpha"_a = std::complex<double>(1.0),
        "beta"_a = std::complex<double>(0.0));
}