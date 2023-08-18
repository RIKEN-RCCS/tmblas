//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "hemm_tmpl.hpp"

namespace tmblas {

template
void hemm<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *B, idx_int ldb,
  std::complex<half> const &beta,
  std::complex<half> *C, idx_int ldc);

template
void hemm<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<double> *C, idx_int ldc);

#ifdef CBLAS_ROUTINES
template<>
void hemm<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<float> *C, idx_int ldc)
{
  cblas_chemm(CblasColMajor,side2cblas(side), uplo2cblas(uplo), (BLAS_INT)m, (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)B, (BLAS_INT)ldb, (BLAS_VOID const *)&beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}

template<>
void hemm<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<double> *C, idx_int ldc)
{
  cblas_zhemm(CblasColMajor,side2cblas(side), uplo2cblas(uplo), (BLAS_INT)m, (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)B, (BLAS_INT)ldb, (BLAS_VOID const *)&beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}
#else

template
void hemm<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<double> *C, idx_int ldc);
#endif

template
void hemm<std::complex<half>, std::complex<half>, std::complex<half> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<half> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *B, idx_int ldb,
  std::complex<half> const &beta,
  std::complex<half> *C, idx_int ldc);

template
void hemm<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<quadruple> const *B, idx_int ldb,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *C, idx_int ldc);

template
void hemm<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<octuple> const &alpha,
  std::complex<octuple> const *A, idx_int lda,
  std::complex<octuple> const *B, idx_int ldb,
  std::complex<octuple> const &beta,
  std::complex<octuple> *C, idx_int ldc);

template
void hemm<std::complex<half>, std::complex<half>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<half> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<half>, std::complex<float>, std::complex<half> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<half> *C, idx_int ldc);

template
void hemm<std::complex<float>, std::complex<half>, std::complex<half> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<half> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<half> *C, idx_int ldc);

template
void hemm<std::complex<half>, std::complex<float>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<half> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<float>, std::complex<half>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<half> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<float>, std::complex<float>, std::complex<half> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<half> *C, idx_int ldc);

template
void hemm<std::complex<float>, std::complex<float>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<double> *C, idx_int ldc);

template
void hemm<std::complex<float>, std::complex<double>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<float>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<double>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<float> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<float>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<double> *C, idx_int ldc);

template
void hemm<std::complex<float>, std::complex<double>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<double> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<quadruple> const *B, idx_int ldb,
  std::complex<quadruple> const &beta,
  std::complex<double> *C, idx_int ldc);

template
void hemm<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<quadruple> const &beta,
  std::complex<double> *C, idx_int ldc);

template
void hemm<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<quadruple> const *B, idx_int ldb,
  std::complex<quadruple> const &beta,
  std::complex<double> *C, idx_int ldc);

template
void hemm<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<quadruple> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *C, idx_int ldc);

template
void hemm<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<quadruple> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<quadruple> const *B, idx_int ldb,
  std::complex<quadruple> const &beta,
  std::complex<quadruple> *C, idx_int ldc);

}
