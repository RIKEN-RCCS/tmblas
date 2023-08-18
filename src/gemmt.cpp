//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "gemmt_tmpl.hpp"

// instantiation of gemmt
// all real half
namespace tmblas {

template
void gemmt<half, half, half, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  half const &alpha,
				  half const *A, idx_int lda,
				  half const *B, idx_int ldb,
				  half const &beta,
				  half *C, idx_int ldc);
// all complex half
template
void gemmt<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<half> const &alpha,
				  std::complex<half> const *A, idx_int lda,
				  std::complex<half> const *B, idx_int ldb,
				  std::complex<half> const &beta,
				  std::complex<half> *C, idx_int ldc);
// all real float

template
void gemmt<float, float, float, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  float const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  float const &beta,
				  float *C, idx_int ldc);
// all complex float
template
void gemmt<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<float> *C, idx_int ldc);

// all real double
template
void gemmt<double, double, double, quadruple>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  double const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  double const &beta,
				  double *C, idx_int ldc);
// all complex double
template
void gemmt<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<double> *C, idx_int ldc);

// instantiation
// specialization to use real C-BLAS not CBLAS_ROUTINES
#ifdef INTEL_MKL
template<> 
void gemmt<float, float, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  float const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  float const &beta,
				  float *C, idx_int ldc)
{
  CBLAS_TRANSPOSE trA, trB;

  trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  trB = (transB == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  
  cblas_sgemmt(CblasColMajor,           // MKL / Veclib C-BLAS 
              blas::uplo2cblas(uplo),
	      trA, trB, (BLAS_INT)n, (BLAS_INT)k, alpha,
	      A, (BLAS_INT)lda, B, (BLAS_INT)ldb, beta, C, (BLAS_INT)ldc ); 
}

template<> 
void gemmt<double, double, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  double const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  double const &beta,
				  double *C, idx_int ldc)
{
  CBLAS_TRANSPOSE trA, trB;

  trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  trB = (transB == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  
  cblas_dgemmt(CblasColMajor,           // MKL / Veclib C-BLAS 
              blas::uplo2cblas(uplo),
	      trA, trB, (BLAS_INT)n, (BLAS_INT)k, alpha,
	      A, (BLAS_INT)lda, B, (BLAS_INT)ldb, beta, C, (BLAS_INT)ldc ); 
}


template<> 
void gemmt<std::complex<float>, std::complex<float>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<float> *C, idx_int ldc)
{
  CBLAS_TRANSPOSE trA, trB;
  if(transA == blas::Op::NoTrans) {
    trA = CblasNoTrans;
  } else if (transA == blas::Op::Trans) {
    trA = CblasTrans;
  }  else {
    trA = CblasConjTrans;
  }
  if(transB == blas::Op::NoTrans) {
    trB = CblasNoTrans;
  } else if (transB == blas::Op::Trans) {
    trB = CblasTrans;
  }  else {
    trB = CblasConjTrans;
  }
//  trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
//  trB = (transB == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  
  cblas_cgemmt(CblasColMajor,           // MKL / Veclib C-BLAS 
              blas::uplo2cblas(uplo),
	      trA, trB, (BLAS_INT)n, k, (BLAS_VOID const  *)&alpha,
	      (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)B, (BLAS_INT)ldb, (BLAS_VOID const *)&beta, (BLAS_VOID *)C, (BLAS_INT)ldc ); 
}

// specialization to use complex C-BLAS

template<> 
void gemmt<std::complex<double>, std::complex<double>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<double> *C, idx_int ldc)
{
  CBLAS_TRANSPOSE trA, trB;

  if(transA == blas::Op::NoTrans) {
    trA = CblasNoTrans;
  } else if (transA == blas::Op::Trans) {
    trA = CblasTrans;
  }  else {
    trA = CblasConjTrans;
  }
  if(transB == blas::Op::NoTrans) {
    trB = CblasNoTrans;
  } else if (transB == blas::Op::Trans) {
    trB = CblasTrans;
  }  else {
    trB = CblasConjTrans;
  }
//  trA = (transA == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
//  trB = (transB == blas::Op::NoTrans ? CblasNoTrans : CblasTrans);
  
  cblas_zgemmt(CblasColMajor,           // MKL / Veclib C-BLAS 
              blas::uplo2cblas(uplo),
	      trA, trB, (BLAS_INT)n, k, (BLAS_VOID const  *)&alpha,
	      (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)B, (BLAS_INT)ldb, (BLAS_VOID const *)&beta, (BLAS_VOID *)C, (BLAS_INT)ldc ); 
}

#else
template
void gemmt<float, float, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  float const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  float const &beta,
				  float *C, idx_int ldc);
template
void gemmt<double, double, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  double const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  double const &beta,
				  double *C, idx_int ldc);

template
void gemmt<std::complex<float>, std::complex<float>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<float> *C, idx_int ldc);

template
void gemmt<std::complex<double>, std::complex<double>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<double> *C, idx_int ldc);
#endif
// instantiation with single real data type

template
void gemmt<half, half, half>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  half const &alpha,
				  half const *A, idx_int lda,
				  half const *B, idx_int ldb,
				  half const &beta,
				  half *C, idx_int ldc);

template
void gemmt<std::complex<half>, std::complex<half>, std::complex<half> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<half> const &alpha,
				  std::complex<half> const *A, idx_int lda,
				  std::complex<half> const *B, idx_int ldb,
				  std::complex<half> const &beta,
				  std::complex<half> *C, idx_int ldc);

template
void gemmt<quadruple, quadruple, quadruple>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  quadruple const &alpha,
				  quadruple const *A, idx_int lda,
				  quadruple const *B, idx_int ldb,
				  quadruple const &beta,
				  quadruple *C, idx_int ldc);

template
void gemmt<octuple, octuple, octuple>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  octuple const &alpha,
				  octuple const *A, idx_int lda,
				  octuple const *B, idx_int ldb,
				  octuple const &beta,
				  octuple *C, idx_int ldc);

// instantiation with single complex data type
template
void gemmt<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<quadruple> const &alpha,
				  std::complex<quadruple> const *A, idx_int lda,
				  std::complex<quadruple> const *B, idx_int ldb,
				  std::complex<quadruple> const &beta,
				  std::complex<quadruple> *C, idx_int ldc);

template
void gemmt<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<octuple> const &alpha,
				  std::complex<octuple> const *A, idx_int lda,
				  std::complex<octuple> const *B, idx_int ldb,
				  std::complex<octuple> const &beta,
				  std::complex<octuple> *C, idx_int ldc);

template
void gemmt<half, half, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  half const *A, idx_int lda,
				  half const *B, idx_int ldb,
				  float const &beta,
				  float *C, idx_int ldc);

template
void gemmt<half, float, half>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  half const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  float const &beta,
				  half *C, idx_int ldc);

template
void gemmt<float, half, half>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  float const *A, idx_int lda,
				  half const *B, idx_int ldb,
				  float const &beta,
				  half *C, idx_int ldc);
template
void gemmt<std::complex<half>, std::complex<half>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<half> const *A, idx_int lda,
				  std::complex<half> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<float> *C, idx_int ldc);

template
void gemmt<std::complex<half>, std::complex<float>, std::complex<half> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<half> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<half> *C, idx_int ldc);

template
void gemmt<std::complex<float>, std::complex<half>, std::complex<half> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<half> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<half> *C, idx_int ldc);
template
void gemmt<half, float, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  half const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  float const &beta,
				  float *C, idx_int ldc);

template
void gemmt<float, half, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  float const *A, idx_int lda,
				  half const *B, idx_int ldb,
				  float const &beta,
				  float *C, idx_int ldc);

template
void gemmt<float, float, half>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  float const &alpha,
				  float const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  float const &beta,
				  half *C, idx_int ldc);

template
void gemmt<std::complex<half>, std::complex<float>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<half> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<float> *C, idx_int ldc);

template
void gemmt<std::complex<float>, std::complex<half>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<half> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<float> *C, idx_int ldc);

template
void gemmt<std::complex<float>, std::complex<float>, std::complex<half> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<float> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<float> const &beta,
				  std::complex<half> *C, idx_int ldc);

// instantiation with mixed real data type (float and double)
// one real double and two real float
template
void gemmt<float, float, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  float const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  double const &beta,
				  double *C, idx_int ldc);

template
void gemmt<float, double, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  float const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  double const &beta,
				  float *C, idx_int ldc);

template
void gemmt<double, float, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  double const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  double const &beta,
				  float *C, idx_int ldc);

// one real double and two complex float
template
void gemmt<std::complex<float>, std::complex<float>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<double> *C, idx_int ldc);

template
void gemmt<std::complex<float>, std::complex<double>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<float> *C, idx_int ldc);

template
void gemmt<std::complex<double>, std::complex<float>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<float> *C, idx_int ldc);

// two real double and one real float
template
void gemmt<double, double, float>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  double const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  double const &beta,
				  float *C, idx_int ldc);

template
void gemmt<double, float, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  double const *A, idx_int lda,
				  float const *B, idx_int ldb,
				  double const &beta,
				  double *C, idx_int ldc);
template
void gemmt<float, double, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  double const &alpha,
				  float const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  double const &beta,
				  double *C, idx_int ldc);

// two complex double and one complex float

template
void gemmt<std::complex<double>, std::complex<double>, std::complex<float> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<float> *C, idx_int ldc);

template
void gemmt<std::complex<double>, std::complex<float>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<float> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<double> *C, idx_int ldc);
template
void gemmt<std::complex<float>, std::complex<double>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<double> const &alpha,
				  std::complex<float> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<double> const &beta,
				  std::complex<double> *C, idx_int ldc);


// instantiation with mixed data type (double and quadruple)
// one real quadruple and two real double
template
void gemmt<double, double, quadruple>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  quadruple const &alpha,
				  double const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  quadruple const &beta,
				  quadruple *C, idx_int ldc);
template
void gemmt<double, quadruple, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  quadruple const &alpha,
				  double const *A, idx_int lda,
				  quadruple const *B, idx_int ldb,
				  quadruple const &beta,
				  double *C, idx_int ldc);
template
void gemmt<quadruple, double, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  quadruple const &alpha,
				  quadruple const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  quadruple const &beta,
				  double *C, idx_int ldc);

// one complex quadruple and two complex double
template
void gemmt<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<quadruple> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<quadruple> const &beta,
				  std::complex<quadruple> *C, idx_int ldc);
template
void gemmt<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<quadruple> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<quadruple> const *B, idx_int ldb,
				  std::complex<quadruple> const &beta,
				  std::complex<double> *C, idx_int ldc);

template
void gemmt<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<quadruple> const &alpha,
				  std::complex<quadruple> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<quadruple> const &beta,
				  std::complex<double> *C, idx_int ldc);

// two real quadruple and one real double

template
void gemmt<quadruple, quadruple, double>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  quadruple const &alpha,
				  quadruple const *A, idx_int lda,
				  quadruple const *B, idx_int ldb,
				  quadruple const &beta,
				  double *C, idx_int ldc);

template
void gemmt<quadruple, double, quadruple>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  quadruple const &alpha,
				  quadruple const *A, idx_int lda,
				  double const *B, idx_int ldb,
				  quadruple const &beta,
				  quadruple *C, idx_int ldc);
template
void gemmt<double, quadruple, quadruple>(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  quadruple const &alpha,
				  double const *A, idx_int lda,
				  quadruple const *B, idx_int ldb,
				  quadruple const &beta,
				  quadruple *C, idx_int ldc);

// two complex quadruple and one complex double

template
void gemmt<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<quadruple> const &alpha,
				  std::complex<quadruple> const *A, idx_int lda,
				  std::complex<quadruple> const *B, idx_int ldb,
				  std::complex<quadruple> const &beta,
				  std::complex<double> *C, idx_int ldc);

template
void gemmt<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<quadruple> const &alpha,
				  std::complex<quadruple> const *A, idx_int lda,
				  std::complex<double> const *B, idx_int ldb,
				  std::complex<quadruple> const &beta,
				  std::complex<quadruple> *C, idx_int ldc);

template
void gemmt<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
                                  blas::Uplo uplo,
				  blas::Op transA,
				  blas::Op transB,
				  idx_int n, idx_int k,
				  std::complex<quadruple> const &alpha,
				  std::complex<double> const *A, idx_int lda,
				  std::complex<quadruple> const *B, idx_int ldb,
				  std::complex<quadruple> const &beta,
   				  std::complex<quadruple> *C, idx_int ldc);

}
