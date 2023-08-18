//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "trsv_tmpl.hpp"

namespace tmblas {

// instantiation of trsv
// all real half

template
void
trsv<half, half, float>(blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  half const *A, const idx_int lda,
	  half *X, const idx_int incx,
	  float *w);
// all complex half
template
void
trsv<std::complex<half>, std::complex<half>, std::complex<float> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<half> const *A, const idx_int lda,
	  std::complex<half> *X,
	  const idx_int incx,
	  std::complex<float> *w);

// all real float
template
void
trsv<float, float, double>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  float const *A, const idx_int lda,
	  float *X, const idx_int incx,
	  double *w);
// all complex float
template
void
trsv<std::complex<float>, std::complex<float>, std::complex<double> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<float> const *A, const idx_int lda,
	  std::complex<float> *X,
	  const idx_int incx,
	  std::complex<double> *w);

// all real double
template
void
trsv<double, double, quadruple>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  double const *A, const idx_int lda,
	  double *X, const idx_int incx,
	  quadruple *w);
// all complex double
template
void
trsv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<double> const *A, const idx_int lda,
	  std::complex<double> *X,
	  const idx_int incx,
	  std::complex<quadruple> *w);

#ifdef CBLAS_ROUTINES
template<>
void
trsv<float, float>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  float const *A, const idx_int lda,
	  float *X, const idx_int incx,
	  float *w)
{
  CBLAS_UPLO uploA;
  CBLAS_TRANSPOSE trA;
  CBLAS_DIAG diagA;
  uploA = (uplo == blas::Uplo::Upper ? CblasUpper : CblasLower);
  trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  diagA = (diag == blas::Diag::NonUnit ? CblasNonUnit : CblasUnit);

  cblas_strsv(CblasColMajor, uploA, trA, diagA, (BLAS_INT)n,
	      A, (BLAS_INT)lda, X, (BLAS_INT)incx);
}
// all complex float
template<>
void
trsv<std::complex<float>, std::complex<float> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<float> const *A, const idx_int lda,
	  std::complex<float> *X,
	  const idx_int incx,
	  std::complex<float> *w)
{
  CBLAS_UPLO uploA;
  CBLAS_TRANSPOSE trA;
  CBLAS_DIAG diagA;
  uploA = (uplo == blas::Uplo::Upper ? CblasUpper : CblasLower);
  //trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  if(transA == blas::Op::NoTrans) {
    trA = CblasNoTrans;
  } else if (transA == blas::Op::Trans) {
    trA = CblasTrans;
  }  else {
    trA = CblasConjTrans;
  }
  diagA = (diag == blas::Diag::NonUnit ? CblasNonUnit : CblasUnit);

  cblas_ctrsv(CblasColMajor, uploA, trA, diagA, (BLAS_INT)n,
	      (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID *)X, (BLAS_INT)incx);
}

// all real double
template<>
void
trsv<double, double>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  double const *A, const idx_int lda,
	  double *X, const idx_int incx,
	  double *w)
{
  
  CBLAS_UPLO uploA;
  CBLAS_TRANSPOSE trA;
  CBLAS_DIAG diagA;
  uploA = (uplo == blas::Uplo::Upper ? CblasUpper : CblasLower);
  trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  diagA = (diag == blas::Diag::NonUnit ? CblasNonUnit : CblasUnit);

  cblas_dtrsv(CblasColMajor, uploA, trA, diagA, (BLAS_INT)n,
	      A, (BLAS_INT)lda, X, (BLAS_INT)incx);

}
// all complex double
template<>
void
trsv<std::complex<double>, std::complex<double> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<double> const *A, const idx_int lda,
	  std::complex<double> *X,
	  const idx_int incx,
	  std::complex<double> *w)
{
  CBLAS_UPLO uploA;
  CBLAS_TRANSPOSE trA;
  CBLAS_DIAG diagA;
  uploA = (uplo == blas::Uplo::Upper ? CblasUpper : CblasLower);
  //trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  if(transA == blas::Op::NoTrans) {
    trA = CblasNoTrans;
  } else if (transA == blas::Op::Trans) {
    trA = CblasTrans;
  }  else {
    trA = CblasConjTrans;
  }
  diagA = (diag == blas::Diag::NonUnit ? CblasNonUnit : CblasUnit);

  cblas_ztrsv(CblasColMajor, uploA, trA, diagA, (BLAS_INT)n,
	      (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID *)X, (BLAS_INT)incx);
}
#else
template
void
trsv<float, float>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  float const *A, const idx_int lda,
	  float *X, const idx_int incx,
	  float *w);
// all complex float
template
void
trsv<std::complex<float>, std::complex<float> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<float> const *A, const idx_int lda,
	  std::complex<float> *X,
	  const idx_int incx,
	  std::complex<float> *w);

// all real double
template
void
trsv<double, double>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  double const *A, const idx_int lda,
	  double *X, const idx_int incx,
	  double  *w);
// all complex double
template
void
trsv<std::complex<double>, std::complex<double> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<double> const *A, const idx_int lda,
	  std::complex<double> *X,
	  const idx_int incx,
	  std::complex<double> *w);
#endif

// all real half
template
void
trsv<half, half>(blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  half const *A, const idx_int lda,
	  half *X, const idx_int incx,
	  half *w);
// all complex half
template
void
trsv<std::complex<half>, std::complex<half> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<half> const *A, const idx_int lda,
	  std::complex<half> *X,
	  const idx_int incx,
	  std::complex<half> *w);

// all real quadruple
template
void
trsv<quadruple, quadruple>(blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  quadruple const *A, const idx_int lda,
	  quadruple *X, const idx_int incx,
	  quadruple *w);
// all complex quadruple
template
void
trsv<std::complex<quadruple>, std::complex<quadruple> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<quadruple> const *A, const idx_int lda,
	  std::complex<quadruple> *X,
	  const idx_int incx,
	  std::complex<quadruple> *w);

// all real octuple
template
void
trsv<octuple, octuple>(blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  octuple const *A, const idx_int lda,
	  octuple *X, const idx_int incx,
	  octuple *w);
// all complex octuple
template
void
trsv<std::complex<octuple>, std::complex<octuple> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<octuple> const *A, const idx_int lda,
	  std::complex<octuple> *X,
	  const idx_int incx,
	  std::complex<octuple> *w);

// instantiation with mixed real data type (float and double)
// real double and real float

template
void
trsv<float, half>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  float const *A, const idx_int lda,
	  half *X, const idx_int incx,
	  float *w);

template
void
trsv<half, float>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  half const *A, const idx_int lda,
	  float *X, const idx_int incx,
	  float *w);

template
void
trsv<std::complex<float>, std::complex<half> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<float> const *A, const idx_int lda,
	  std::complex<half> *X, const idx_int incx,
	  std::complex<float> *w);

template
void
trsv<std::complex<half>, std::complex<float> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<half> const *A, const idx_int lda,
	  std::complex<float> *X, const idx_int incx,
	  std::complex<float> *w);

template
void
trsv<double, float>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  double const *A, const idx_int lda,
	  float *X, const idx_int incx,
	  double *w);
template
void
trsv<float, double>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  float const *A, const idx_int lda,
	  double *X, const idx_int incx,
	  double *w);
// all complex float
template
void
trsv<std::complex<double>, std::complex<float> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<double> const *A, const idx_int lda,
	  std::complex<float> *X,
	  const idx_int incx,
	  std::complex<double> *w);
template
void
trsv<std::complex<float>, std::complex<double> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<float> const *A, const idx_int lda,
	  std::complex<double> *X,
	  const idx_int incx,
	  std::complex<double> *w);

// all real double
template
void
trsv<quadruple, double>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  quadruple const *A, const idx_int lda,
	  double *X, const idx_int incx,
	  quadruple *w);
template
void
trsv<double, quadruple>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  double const *A, const idx_int lda,
	  quadruple *X, const idx_int incx,
	  quadruple *w);
// all complex double
template
void
trsv<std::complex<quadruple>, std::complex<double> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<quadruple> const *A, const idx_int lda,
	  std::complex<double> *X,
	  const idx_int incx,
	  std::complex<quadruple> *w);
template
void
trsv<std::complex<double>, std::complex<quadruple> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<double> const *A, const idx_int lda,
	  std::complex<quadruple> *X,
	  const idx_int incx,
	  std::complex<quadruple> *w);

}

