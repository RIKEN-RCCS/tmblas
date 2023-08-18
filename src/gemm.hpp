//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _GEMMP_HPP
# define _GEMMP_HPP

namespace tmblas {

// =============================================================================
/// General matrix-matrix multiply:
/// \[
///     C = \alpha op(A) \times op(B) + \beta C,
/// \]
/// where $op(X)$ is one of
///     $op(X) = X$,
///     $op(X) = X^T$, or
///     $op(X) = X^H$,
/// alpha and beta are scalars, and A, B, and C are matrices, with
/// $op(A)$ an m-by-k matrix, $op(B)$ a k-by-n matrix, and C an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] transA
///     The operation $op(A)$ to be used:
///     - Op::NoTrans:   $op(A) = A$.
///     - Op::Trans:     $op(A) = A^T$.
///     - Op::ConjTrans: $op(A) = A^H$.
///
/// @param[in] transB
///     The operation $op(B)$ to be used:
///     - Op::NoTrans:   $op(B) = B$.
///     - Op::Trans:     $op(B) = B^T$.
///     - Op::ConjTrans: $op(B) = B^H$.
///
/// @param[in] m
///     Number of rows of the matrix C and $op(A)$. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix C and $op(B)$. n >= 0.
///
/// @param[in] k
///     Number of columns of $op(A)$ and rows of $op(B)$. k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If transA = NoTrans:
///       the m-by-k matrix A, stored in an lda-by-k array [RowMajor: m-by-lda].
///     - Otherwise:
///       the k-by-m matrix A, stored in an lda-by-m array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If transA = NoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
///     - Otherwise:           lda >= max(1, k) [RowMajor: lda >= max(1, m)].
///
/// @param[in] B
///     - If transB = NoTrans:
///       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
///     - Otherwise:
///       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If transB = NoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
///     - Otherwise:           ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C, stored in an ldc-by-n array [RowMajor: m-by-ldc].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
///
/// @ingroup gemm

template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> >
void gemm(blas::Op transA,
           blas::Op transB,
          int m, int n, int k,
          blas::scalar_type<Ta, Tb, Tc> const &alpha,
          Ta const *A, int lda,
          Tb const *B, int ldb,
          blas::scalar_type<Ta, Tb, Tc> const &beta,
          Tc *C, int ldc);

#ifdef CBLAS_ROUTINES
template<> 
void gemm<float, float, float>(
				  blas::Op transA,
				  blas::Op transB,
				  idx_int m, idx_int n, idx_int k, 
				  float const &alpha, 
				  float const *A, idx_int lda, 
				  float const *B, idx_int ldb, 
				  float const &beta, 
				  float *C, idx_int ldc);

template<> 
void gemm<double, double, double>(
				  blas::Op transA,
				  blas::Op transB,
				  idx_int m, idx_int n, idx_int k, 
				  double const &alpha, 
				  double const *A, idx_int lda, 
				  double const *B, idx_int ldb, 
				  double const &beta, 
				  double *C, idx_int ldc);

template<> 
void gemm<std::complex<float>, std::complex<float>, std::complex<float> >(
				  blas::Op transA,
				  blas::Op transB,
				  idx_int m, idx_int n, idx_int k, 
				  std::complex<float> const &alpha, 
				  std::complex<float> const *A, idx_int lda, 
				  std::complex<float> const *B, idx_int ldb, 
				  std::complex<float> const &beta, 
				  std::complex<float> *C, idx_int ldc);

template<> 
void gemm<std::complex<double>, std::complex<double>, std::complex<double> >(
				  blas::Op transA,
				  blas::Op transB,
				  idx_int m, idx_int n, idx_int k, 
				  std::complex<double> const &alpha, 
				  std::complex<double> const *A, idx_int lda, 
				  std::complex<double> const *B, idx_int ldb, 
				  std::complex<double> const &beta, 
				  std::complex<double> *C, idx_int ldc);
#endif
}

#endif
