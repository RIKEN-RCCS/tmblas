//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "geru_tmpl.hpp"

namespace tmblas{

template
void geru<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void geru<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

#ifdef CBLAS_ROUTINES
template<>
void geru<std::complex<float>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda)
{
  cblas_cgeru(CblasColMajor, m, n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)x, incx, (BLAS_VOID const *)y, incy, (BLAS_VOID *)A, lda);
}

template<>
void geru<std::complex<double>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda)
{
  cblas_zgeru(CblasColMajor, m, n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)x, incx, (BLAS_VOID const *)y, incy, (BLAS_VOID *)A, lda);
}
#else
template
void geru<std::complex<float>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);
#endif
  
template
void geru<std::complex<half>, std::complex<half>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void geru<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

template
void geru<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
  idx_int m, idx_int n,
  std::complex<octuple> const &alpha,
  std::complex<octuple> const *x, idx_int incx,
  std::complex<octuple> const *y, idx_int incy,
  std::complex<octuple> *A, idx_int lda);

template
void geru<std::complex<half>, std::complex<half>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<half>, std::complex<float>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void geru<std::complex<float>, std::complex<half>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void geru<std::complex<half>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<float>, std::complex<half>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<half> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<float>, std::complex<float>, std::complex<half> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<half> *A, idx_int lda);

template
void geru<std::complex<float>, std::complex<float>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void geru<std::complex<float>, std::complex<double>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<double>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<float>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void geru<std::complex<float>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void geru<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void geru<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);

template
void geru<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

template
void geru<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const *y, idx_int incy,
  std::complex<quadruple> *A, idx_int lda);

}

