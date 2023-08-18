//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "hemv_tmpl.hpp"

namespace tmblas{
// internal higher precision
// HHHF
template
void hemv<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);

// FFFD  
template
void hemv<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

 // DDDQ
template
void hemv<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

// HHH
template
void hemv<std::complex<half>, std::complex<half>, std::complex<half> > (
  blas::Uplo uplo,
  idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<half> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<half> *w);

// mixed precision amang input data
// FFF
#ifdef CBLAS_ROUTINES
template<>
void hemv<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w)
{
  cblas_chemv(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)x, (BLAS_INT)incx, (BLAS_VOID const *)&beta, (BLAS_VOID *)y, (BLAS_INT)incy);
}
// DDD
template<>
void hemv<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w)
{
  cblas_zhemv(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, (BLAS_VOID const  *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)x, (BLAS_INT)incx, (BLAS_VOID const  *)&beta, (BLAS_VOID *)y, (BLAS_INT)incy);
}
#else
// FFF
template
void hemv<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);
// DDD
template
void hemv<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);
#endif

// QQQ
template
void hemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);
// OOO
template
void hemv<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<octuple> const &alpha,
  std::complex<octuple> const *A, idx_int lda,
  std::complex<octuple> const *x, idx_int incx,
  std::complex<octuple> const &beta,
  std::complex<octuple> *y, idx_int incy, std::complex<octuple> *w);

// HHF
template
void hemv<std::complex<half>, std::complex<half>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

// HFH
template
void hemv<std::complex<half>, std::complex<float>, std::complex<half> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);

// FHH
template
void hemv<std::complex<float>, std::complex<half>, std::complex<half> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);

// HFF
template
void hemv<std::complex<half>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

// FHF
template
void hemv<std::complex<float>, std::complex<half>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<half> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<float> *w);

// FFH
template
void hemv<std::complex<float>, std::complex<float>, std::complex<half> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const &beta,
  std::complex<half> *y, idx_int incy, std::complex<float> *w);

// FFD
template
void hemv<std::complex<float>, std::complex<float>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);

// FDF
template
void hemv<std::complex<float>, std::complex<double>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

// DFF
template
void hemv<std::complex<double>, std::complex<float>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

// DDF
template
void hemv<std::complex<double>, std::complex<double>, std::complex<float> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<float> *y, idx_int incy, std::complex<double> *w);

// DFD
template
void hemv<std::complex<double>, std::complex<float>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<float> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);

// FDD
template
void hemv<std::complex<float>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<double> *w);

// DDQ
template
void hemv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);

// DQD
template
void hemv<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

// QDD
template
void hemv<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

// QQD
template
void hemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<double> *y, idx_int incy, std::complex<quadruple> *w);

// QDQ
template
void hemv<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<double> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);

// DQQ
template
void hemv<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
  blas::Uplo uplo,
  idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<quadruple> const *x, idx_int incx,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *y, idx_int incy, std::complex<quadruple> *w);

}

