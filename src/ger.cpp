//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "ger_tmpl.hpp"

namespace tmblas{

template
void ger<half, half, half, float>(
  idx_int m, idx_int n,
  half const &alpha,
  half const *x, idx_int incx,
  half const *y, idx_int incy,
  half *A, idx_int lda);

template
void ger<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void ger<float, float, float, double>(
  idx_int m, idx_int n,
  float const &alpha,
  float const *x, idx_int incx,
  float const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<double, double, double, quadruple>(
  idx_int m, idx_int n,
  double const &alpha,
  double const *x, idx_int incx,
  double const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

#ifdef CBLAS_ROUTINES
template<>
void ger<float, float, float>(
  idx_int m, idx_int n,
  float const &alpha,
  float const *x, idx_int incx,
  float const *y, idx_int incy,
  float *A, idx_int lda)
{
  cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template<>
void ger<double, double, double>(
  idx_int m, idx_int n,
  double const &alpha,
  double const *x, idx_int incx,
  double const *y, idx_int incy,
  double *A, idx_int lda)
{
  cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template<>
void ger<std::complex<float>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda)
{
  cblas_cgerc(CblasColMajor, m, n, (BLAS_VOID const  *)&alpha, (BLAS_VOID const *)x, incx, (BLAS_VOID const *)y, incy, (BLAS_VOID *)A, lda);
}

template<>
void ger<std::complex<double>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda)
{
  cblas_zgerc(CblasColMajor, m, n, (BLAS_VOID const  *)&alpha, (BLAS_VOID const *)x, incx, (BLAS_VOID const *)y, incy, (BLAS_VOID *)A, lda);
}
#else
template
void ger<float, float, float>(
  idx_int m, idx_int n,
  float const &alpha,
  float const *x, idx_int incx,
  float const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<double, double, double>(
  idx_int m, idx_int n,
  double const &alpha,
  double const *x, idx_int incx,
  double const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);
#endif
  
template
void ger<half, half, half>(
  idx_int m, idx_int n,
  half const &alpha,
  half const *x, idx_int incx,
  half const *y, idx_int incy,
  half *A, idx_int lda);

template
void ger<std::complex<half>, std::complex<half>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void ger<quadruple, quadruple, quadruple>(
  idx_int m, idx_int n,
  quadruple const &alpha,
  quadruple const *x, idx_int incx,
  quadruple const *y, idx_int incy,
  quadruple *A, idx_int lda);

template
void ger<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

template
void ger<octuple, octuple, octuple>(
  idx_int m, idx_int n,
  octuple const &alpha,
  octuple const *x, idx_int incx,
  octuple const *y, idx_int incy,
  octuple *A, idx_int lda);

template
void ger<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
  idx_int m, idx_int n,
  std::complex<octuple> const &alpha,
  std::complex<octuple> const *x, idx_int incx,
  std::complex<octuple> const *y, idx_int incy,
  std::complex<octuple> *A, idx_int lda);

template
void ger<half, half, float>(
  idx_int m, idx_int n,
  float const &alpha,
  half const *x, idx_int incx,
  half const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<half, float, half>(
  idx_int m, idx_int n,
  float const &alpha,
  half const *x, idx_int incx,
  float const *y, idx_int incy,
  half *A, idx_int lda);

template
void ger<float, half, half>(
  idx_int m, idx_int n,
  float const &alpha,
  float const *x, idx_int incx,
  half const *y, idx_int incy,
  half *A, idx_int lda);

template
void ger<std::complex<half>, std::complex<half>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<std::complex<half>, std::complex<float>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<half>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void ger<half, float, float>(
  idx_int m, idx_int n,
  float const &alpha,
  half const *x, idx_int incx,
  float const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<float, half, float>(
  idx_int m, idx_int n,
  float const &alpha,
  float const *x, idx_int incx,
  half const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<float, float, half>(
  idx_int m, idx_int n,
  float const &alpha,
  float const *x, idx_int incx,
  float const *y, idx_int incy,
  half *A, idx_int lda);

template
void ger<std::complex<half>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<half>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<float>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void ger<float, float, double>(
  idx_int m, idx_int n,
  double const &alpha,
  float const *x, idx_int incx,
  float const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<float, double, float>(
  idx_int m, idx_int n,
  double const &alpha,
  float const *x, idx_int incx,
  double const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<double, float, float>(
  idx_int m, idx_int n,
  double const &alpha,
  double const *x, idx_int incx,
  float const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<float>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<double>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<double, double, float>(
  idx_int m, idx_int n,
  double const &alpha,
  double const *x, idx_int incx,
  double const *y, idx_int incy,
  float *A, idx_int lda);

template
void ger<double, float, double>(
  idx_int m, idx_int n,
  double const &alpha,
  double const *x, idx_int incx,
  float const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<float, double, double>(
  idx_int m, idx_int n,
  double const &alpha,
  float const *x, idx_int incx,
  double const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<double>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<float>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void ger<std::complex<float>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void ger<double, double, quadruple>(
  idx_int m, idx_int n,
  quadruple const &alpha,
  double const *x, idx_int incx,
  double const *y, idx_int incy,
  quadruple *A, idx_int lda);

template
void ger<double, quadruple, double>(
  idx_int m, idx_int n,
  quadruple const &alpha,
  double const *x, idx_int incx,
  quadruple const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<quadruple, double, double>(
  idx_int m, idx_int n,
  quadruple const &alpha,
  quadruple const *x, idx_int incx,
  double const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void ger<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void ger<quadruple, quadruple, double>(
  idx_int m, idx_int n,
  quadruple const &alpha,
  quadruple const *x, idx_int incx,
  quadruple const *y, idx_int incy,
  double *A, idx_int lda);

template
void ger<quadruple, double, quadruple>(
  idx_int m, idx_int n,
  quadruple const &alpha,
  quadruple const *x, idx_int incx,
  double const *y, idx_int incy,
  quadruple *A, idx_int lda);

template
void ger<double, quadruple, quadruple>(
  idx_int m, idx_int n,
  quadruple const &alpha,
  double const *x, idx_int incx,
  quadruple const *y, idx_int incy,
  quadruple *A, idx_int lda);

template
void ger<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void ger<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

template
void ger<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

}

