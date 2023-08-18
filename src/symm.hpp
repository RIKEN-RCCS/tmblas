//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SYMM_HPP
#define _SYMM_HPP

namespace tmblas {

// =============================================================================
/// Symmetric matrix-matrix multiply:
/// \[
///     C = \alpha A B + \beta C,
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C,
/// \]
/// where alpha and beta are scalars, A is an m-by-m or n-by-n symmetric matrix,
/// and B and C are m-by-n matrices.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] side
///     The side the matrix A appears on:
///     - Side::Left:  $C = \alpha A B + \beta C$,
///     - Side::Right: $C = \alpha B A + \beta C$.
///
/// @param[in] uplo
///     What part of the matrix A is referenced:
///     - Uplo::Lower: only the lower triangular part of A is referenced.
///     - Uplo::Upper: only the upper triangular part of A is referenced.
///
/// @param[in] m
///     Number of rows of the matrices B and C.
///
/// @param[in] n
///     Number of columns of the matrices B and C.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If side = Left:  The m-by-m matrix A, stored in an lda-by-m array.
///     - If side = Right: The n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of A.
///     - If side = Left:  lda >= max(1, m).
///     - If side = Right: lda >= max(1, n).
///
/// @param[in] B
///     The m-by-n matrix B, stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, n).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C, stored in an lda-by-n array.
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, n).
///
/// @ingroup symm

template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> >
void symm(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  blas::scalar_type<Ta, Tb, Tc> const &alpha,
  Ta const *A, idx_int lda,
  Tb const *B, idx_int ldb,
  blas::scalar_type<Ta, Tb, Tc> const &beta,
  Tc *C, idx_int ldc);

#ifdef CBLAS_ROUTINES
template<>
void symm<float, float, float>(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  float const *B, idx_int ldb,
  float const &beta,
  float *C, idx_int ldc);

template<>
void symm<std::complex<float>, std::complex<float>, std::complex<float> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *A, idx_int lda,
  std::complex<float> const *B, idx_int ldb,
  std::complex<float> const &beta,
  std::complex<float> *C, idx_int ldc);

template<>
void symm<double, double, double>(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  double const *B, idx_int ldb,
  double const &beta,
  double *C, idx_int ldc);

template<>
void symm<std::complex<double>, std::complex<double>, std::complex<double> >(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *A, idx_int lda,
  std::complex<double> const *B, idx_int ldb,
  std::complex<double> const &beta,
  std::complex<double> *C, idx_int ldc);

#endif

}

#endif
