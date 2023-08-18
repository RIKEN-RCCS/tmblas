//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "gemv_tmpl.hpp"

namespace tmblas {

template
void gemv<half, half, half, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     half const &alpha,
     half const *A, idx_int lda,
     half const *x, idx_int incx,
     half const &beta,
     half *y, idx_int incy);

template
void gemv<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<half> const &alpha,
     std::complex<half> const *A, idx_int lda,
     std::complex<half> const *x, idx_int incx,
     std::complex<half> const &beta,
     std::complex<half> *y, idx_int incy);

template
void gemv<float, float, float, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     float const *A, idx_int lda,
     float const *x, idx_int incx,
     float const &beta,
     float *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<double, double, double, quadruple>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     double const *A, idx_int lda,
     double const *x, idx_int incx,
     double const &beta,
     double *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<double> *y, idx_int incy);

#ifdef CBLAS_ROUTINES
template<>
void gemv<float, float, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     float const *A, idx_int lda,
     float const *x, idx_int incx,
     float const &beta,
     float *y, idx_int incy)
{
  cblas_sgemv(CblasColMajor, op2cblas(trans), (BLAS_INT)m, (BLAS_INT)n, alpha, A, lda, x, (BLAS_INT)incx, beta, y, (BLAS_INT)incy);
}

template<>
void gemv<double, double, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     double const *A, idx_int lda,
     double const *x, idx_int incx,
     double const &beta,
     double *y, idx_int incy)
{
  cblas_dgemv(CblasColMajor, op2cblas(trans), (BLAS_INT)m, (BLAS_INT)n, alpha, A, lda, x, (BLAS_INT)incx, beta, y, (BLAS_INT)incy);
}

template<>
void gemv<std::complex<float>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<float> *y, idx_int incy)
{
  cblas_cgemv(CblasColMajor, op2cblas(trans), (BLAS_INT)m, (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, lda, (BLAS_VOID const *)x, (BLAS_INT)incx, (BLAS_VOID const  *)&beta, (BLAS_VOID *)y, (BLAS_INT)incy);
}

template<>
void gemv<std::complex<double>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<double> *y, idx_int incy)
{
  cblas_zgemv(CblasColMajor, op2cblas(trans), (BLAS_INT)m, (BLAS_INT)n, (BLAS_VOID const  *)&alpha, (BLAS_VOID const *)A, lda, (BLAS_VOID const *)x, (BLAS_INT)incx, (BLAS_VOID const *)&beta, (BLAS_VOID *)y, (BLAS_INT)incy);
}
#else
template
void gemv<float, float, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     float const *A, idx_int lda,
     float const *x, idx_int incx,
     float const &beta,
     float *y, idx_int incy);

template
void gemv<double, double, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     double const *A, idx_int lda,
     double const *x, idx_int incx,
     double const &beta,
     double *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<double> *y, idx_int incy);
#endif

template
void gemv<half, half, half>(
     blas::Op trans,
     idx_int m, idx_int n,
     half const &alpha,
     half const *A, idx_int lda,
     half const *x, idx_int incx,
     half const &beta,
     half *y, idx_int incy);

template
void gemv<std::complex<half>, std::complex<half>, std::complex<half> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<half> const &alpha,
     std::complex<half> const *A, idx_int lda,
     std::complex<half> const *x, idx_int incx,
     std::complex<half> const &beta,
     std::complex<half> *y, idx_int incy);

template
void gemv<quadruple, quadruple, quadruple>(
     blas::Op trans,
     idx_int m, idx_int n,
     quadruple const &alpha,
     quadruple const *A, idx_int lda,
     quadruple const *x, idx_int incx,
     quadruple const &beta,
     quadruple *y, idx_int incy);

template
void gemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<quadruple> const &alpha,
     std::complex<quadruple> const *A, idx_int lda,
     std::complex<quadruple> const *x, idx_int incx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y, idx_int incy);

template
void gemv<octuple, octuple, octuple>(
     blas::Op trans,
     idx_int m, idx_int n,
     octuple const &alpha,
     octuple const *A, idx_int lda,
     octuple const *x, idx_int incx,
     octuple const &beta,
     octuple *y, idx_int incy);

template
void gemv<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<octuple> const &alpha,
     std::complex<octuple> const *A, idx_int lda,
     std::complex<octuple> const *x, idx_int incx,
     std::complex<octuple> const &beta,
     std::complex<octuple> *y, idx_int incy);

template
void gemv<half, half, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     half const *A, idx_int lda,
     half const *x, idx_int incx,
     float const &beta,
     float *y, idx_int incy);

template
void gemv<half, float, half>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     half const *A, idx_int lda,
     float const *x, idx_int incx,
     float const &beta,
     half *y, idx_int incy);

template
void gemv<float, half, half>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     float const *A, idx_int lda,
     half const *x, idx_int incx,
     float const &beta,
     half *y, idx_int incy);

template
void gemv<std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<half> const *A, idx_int lda,
     std::complex<half> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<std::complex<half>, std::complex<float>, std::complex<half> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<half> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<half> *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<half>, std::complex<half> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<half> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<half> *y, idx_int incy);

template
void gemv<half, float, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     half const *A, idx_int lda,
     float const *x, idx_int incx,
     float const &beta,
     float *y, idx_int incy);

template
void gemv<float, half, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     float const *A, idx_int lda,
     half const *x, idx_int incx,
     float const &beta,
     float *y, idx_int incy);

template
void gemv<float, float, half>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     float const *A, idx_int lda,
     float const *x, idx_int incx,
     float const &beta,
     half *y, idx_int incy);

template
void gemv<std::complex<half>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<half> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<half>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<half> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<float>, std::complex<half> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<half> *y, idx_int incy);

template
void gemv<float, float, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     float const *A, idx_int lda,
     float const *x, idx_int incx,
     double const &beta,
     double *y, idx_int incy);

template
void gemv<float, double, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     float const *A, idx_int lda,
     double const *x, idx_int incx,
     double const &beta,
     float *y, idx_int incy);

template
void gemv<double, float, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     double const *A, idx_int lda,
     float const *x, idx_int incx,
     double const &beta,
     float *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<double> *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<double>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<double, double, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     double const *A, idx_int lda,
     double const *x, idx_int incx,
     double const &beta,
     float *y, idx_int incy);

template
void gemv<double, float, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     double const *A, idx_int lda,
     float const *x, idx_int incx,
     double const &beta,
     double *y, idx_int incy);

template
void gemv<float, double, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     float const *A, idx_int lda,
     double const *x, idx_int incx,
     double const &beta,
     double *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<double>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<float> *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<float>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<double> *y, idx_int incy);

template
void gemv<std::complex<float>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<double> *y, idx_int incy);

template
void gemv<double, double, quadruple>(
     blas::Op trans,
     idx_int m, idx_int n,
     quadruple const &alpha,
     double const *A, idx_int lda,
     double const *x, idx_int incx,
     quadruple const &beta,
     quadruple *y, idx_int incy);

template
void gemv<double, quadruple, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     quadruple const &alpha,
     double const *A, idx_int lda,
     quadruple const *x, idx_int incx,
     quadruple const &beta,
     double *y, idx_int incy);

template
void gemv<quadruple, double, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     quadruple const &alpha,
     quadruple const *A, idx_int lda,
     double const *x, idx_int incx,
     quadruple const &beta,
     double *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<quadruple> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<quadruple> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<quadruple> const *x, idx_int incx,
     std::complex<quadruple> const &beta,
     std::complex<double> *y, idx_int incy);

template
void gemv<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<quadruple> const &alpha,
     std::complex<quadruple> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<quadruple> const &beta,
     std::complex<double> *y, idx_int incy);

template
void gemv<quadruple, quadruple, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     quadruple const &alpha,
     quadruple const *A, idx_int lda,
     quadruple const *x, idx_int incx,
     quadruple const &beta,
     double *y, idx_int incy);

template
void gemv<quadruple, double, quadruple>(
     blas::Op trans,
     idx_int m, idx_int n,
     quadruple const &alpha,
     quadruple const *A, idx_int lda,
     double const *x, idx_int incx,
     quadruple const &beta,
     quadruple *y, idx_int incy);

template
void gemv<double, quadruple, quadruple>(
     blas::Op trans,
     idx_int m, idx_int n,
     quadruple const &alpha,
     double const *A, idx_int lda,
     quadruple const *x, idx_int incx,
     quadruple const &beta,
     quadruple *y, idx_int incy);

template
void gemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<quadruple> const &alpha,
     std::complex<quadruple> const *A, idx_int lda,
     std::complex<quadruple> const *x, idx_int incx,
     std::complex<quadruple> const &beta,
     std::complex<double> *y, idx_int incy);

template
void gemv<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<quadruple> const &alpha,
     std::complex<quadruple> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y, idx_int incy);

template
void gemv<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<quadruple> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<quadruple> const *x, idx_int incx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y, idx_int incy);

}

