//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "symv_tmpl.hpp"

namespace tmblas{

template
void symv<half, half, half, float>(
  blas::Uplo uplo,
  idx_int n,
  half const &alpha,
  half const *A, idx_int lda,
  half const *x, idx_int incx,
  half const &beta,
  half *y, idx_int incy, float *w);

template
void symv<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);

template
void symv<float, float, float, double>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  float const *x, idx_int incx,
  float const &beta,
  float *y, idx_int incy, double *w);

template
void symv<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

template
void symv<double, double, double, quadruple>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  double const *x, idx_int incx,
  double const &beta,
  double *y, idx_int incy, quadruple *w);

template
void symv<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

#ifdef CBLAS_ROUTINES
template<>
void symv<float, float, float>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  float const *x, idx_int incx,
  float const &beta,
  float *y, idx_int incy, float *w)
{
  cblas_ssymv(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, alpha, A, (BLAS_INT)lda, x, (BLAS_INT)incx, beta, y, (BLAS_INT)incy);
}

template<>
void symv<double, double, double>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  double const *x, idx_int incx,
  double const &beta,
  double *y, idx_int incy, double *w)
{
  cblas_dsymv(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, alpha, A, (BLAS_INT)lda, x, (BLAS_INT)incx, beta, y, (BLAS_INT)incy);
}

/* Hermitian version will be implemented as different routine hemv
template<>
void symv<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w)
{
  cblas_chemv(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, (BLAS_VOID const *) &alpha, A, (BLAS_INT)lda, x, (BLAS_INT)incx, (BLAS_VOID const *) &beta, y, (BLAS_INT)incy);
}

template<>
void symv<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w)
{
  cblas_zhemv(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, (BLAS_VOID const *) &alpha, A, (BLAS_INT)lda, x, (BLAS_INT)incx, (BLAS_VOID const *) &beta, y, (BLAS_INT)incy);
}
*/
template
void symv<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

template
void symv<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);

#else
template
void symv<float, float, float>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  float const *x, idx_int incx,
  float const &beta,
  float *y, idx_int incy, float *w);

template
void symv<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

template
void symv<double, double, double>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  double const *x, idx_int incx,
  double const &beta,
  double *y, idx_int incy, double *w);

template
void symv<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);
#endif

template
void symv<half, half, half>(
  blas::Uplo uplo,
  idx_int n,
  half const &alpha,
  half const *A, idx_int lda,
  half const *x, idx_int incx,
  half const &beta,
  half *y, idx_int incy, half *w);

template
void symv<std::complex<half>, std::complex<half>, std::complex<half> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<half> *w);

template
void symv<quadruple, quadruple, quadruple>(
  blas::Uplo uplo,
  idx_int n,
  quadruple const &alpha,
  quadruple const *A, idx_int lda,
  quadruple const *x, idx_int incx,
  quadruple const &beta,
  quadruple *y, idx_int incy, quadruple *w);

template
void symv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);

template
void symv<octuple, octuple, octuple>(
  blas::Uplo uplo,
  idx_int n,
  octuple const &alpha,
  octuple const *A, idx_int lda,
  octuple const *x, idx_int incx,
  octuple const &beta,
  octuple *y, idx_int incy, octuple *w);

template
void symv<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<octuple> const &alpha,
  std::complex<octuple> const *A, idx_int lda,
  std::complex<octuple> const *x, idx_int incx,
  std::complex<octuple> const &beta,
  std::complex<octuple> *y, idx_int incy, std::complex<octuple> *w);

template
void symv<half, half, float>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  half const *A, idx_int lda,
  half const *x, idx_int incx,
  float const &beta,
  float *y, idx_int incy, float *w);

template
void symv<half, float, half>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  half const *A, idx_int lda,
  float const *x, idx_int incx,
  float const &beta,
  half *y, idx_int incy, float *w);

template
void symv<float, half, half>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  half const *x, idx_int incx,
  float const &beta,
  half *y, idx_int incy, float *w);

template
void symv<std::complex<half>, std::complex<half>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

template
void symv<std::complex<half>, std::complex<float>, std::complex<half> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);

template
void symv<std::complex<float>, std::complex<half>, std::complex<half> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);

template
void symv<half, float, float>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  half const *A, idx_int lda,
  float const *x, idx_int incx,
  float const &beta,
  float *y, idx_int incy, float *w);

template
void symv<float, half, float>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  half const *x, idx_int incx,
  float const &beta,
  float *y, idx_int incy, float *w);

template
void symv<float, float, half>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  float const *x, idx_int incx,
  float const &beta,
  half *y, idx_int incy, float *w);

template
void symv<std::complex<half>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

template
void symv<std::complex<float>, std::complex<half>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

template
void symv<std::complex<float>, std::complex<float>, std::complex<half> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);


template
void symv<float, float, double>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  float const *A, idx_int lda,
  float const *x, idx_int incx,
  double const &beta,
  double *y, idx_int incy, double *w);

template
void symv<float, double, float>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  float const *A, idx_int lda,
  double const *x, idx_int incx,
  double const &beta,
  float *y, idx_int incy, double *w);

template
void symv<double, float, float>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  float const *x, idx_int incx,
  double const &beta,
  float *y, idx_int incy, double *w);

template
void symv<std::complex<float>, std::complex<float>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);

template
void symv<std::complex<float>, std::complex<double>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

template
void symv<std::complex<double>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

template
void symv<double, double, float>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  double const *x, idx_int incx,
  double const &beta,
  float *y, idx_int incy, double *w);

template
void symv<double, float, double>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  float const *x, idx_int incx,
  double const &beta,
  double *y, idx_int incy, double *w);

template
void symv<float, double, double>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  float const *A, idx_int lda,
  double const *x, idx_int incx,
  double const &beta,
  double *y, idx_int incy, double *w);

template
void symv<std::complex<double>, std::complex<double>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

template
void symv<std::complex<double>, std::complex<float>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);

template
void symv<std::complex<float>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);

template
void symv<double, double, quadruple>(
  blas::Uplo uplo,
  idx_int n,
  quadruple const &alpha,
  double const *A, idx_int lda,
  double const *x, idx_int incx,
  quadruple const &beta,
  quadruple *y, idx_int incy, quadruple *w);

template
void symv<double, quadruple, double>(
  blas::Uplo uplo,
  idx_int n,
  quadruple const &alpha,
  double const *A, idx_int lda,
  quadruple const *x, idx_int incx,
  quadruple const &beta,
  double *y, idx_int incy, quadruple *w);

template
void symv<quadruple, double, double>(
  blas::Uplo uplo,
  idx_int n,
  quadruple const &alpha,
  quadruple const *A, idx_int lda,
  double const *x, idx_int incx,
  quadruple const &beta,
  double *y, idx_int incy, quadruple *w);

template
void symv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);

template
void symv<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

template
void symv<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

template
void symv<quadruple, quadruple, double>(
  blas::Uplo uplo,
  idx_int n,
  quadruple const &alpha,
  quadruple const *A, idx_int lda,
  quadruple const *x, idx_int incx,
  quadruple const &beta,
  double *y, idx_int incy, quadruple *w);

template
void symv<quadruple, double, quadruple>(
  blas::Uplo uplo,
  idx_int n,
  quadruple const &alpha,
  quadruple const *A, idx_int lda,
  double const *x, idx_int incx,
  quadruple const &beta,
  quadruple *y, idx_int incy, quadruple *w);

template
void symv<double, quadruple, quadruple>(
  blas::Uplo uplo,
  idx_int n,
  quadruple const &alpha,
  double const *A, idx_int lda,
  quadruple const *x, idx_int incx,
  quadruple const &beta,
  quadruple *y, idx_int incy, quadruple *w);

template
void symv<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

template
void symv<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);

template
void symv<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);

}

