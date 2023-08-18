//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TRSM_HPP
#define _TRSM_HPP

namespace tmblas{

// =============================================================================
/// Solve the triangular matrix-vector equation
/// \[
///     op(A) X = \alpha B,
/// \]
/// or
/// \[
///     X op(A) = \alpha B,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// X and B are m-by-n matrices, and A is an m-by-m or n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// @see latrs for a more numerically robust implementation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] side
///     Whether $op(A)$ is on the left or right of X:
///     - Side::Left:  $op(A) X = B$.
///     - Side::Right: $X op(A) = B$.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero:
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///
/// @param[in] trans
///     The form of $op(A)$:
///     - Op::NoTrans:   $op(A) = A$.
///     - Op::Trans:     $op(A) = A^T$.
///     - Op::ConjTrans: $op(A) = A^H$.
///
/// @param[in] diag
///     Whether A has a unit or non-unit diagonal:
///     - Diag::Unit:    A is assumed to be unit triangular.
///     - Diag::NonUnit: A is not assumed to be unit triangular.
///
/// @param[in] m
///     Number of rows of matrices B and X. m >= 0.
///
/// @param[in] n
///     Number of columns of matrices B and X. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not accessed.
///
/// @param[in] A
///     - If side = Left:
///       the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
///     - If side = Right:
///       the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If side = left:  lda >= max(1, m).
///     - If side = right: lda >= max(1, n).
///
/// @param[in, out] B
///     On entry,
///     the m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
///     On exit, overwritten by the solution matrix X.
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
///
/// @param w
///     Work array of size matsize where matsize = n*m if side == Side::Right && (trans==Op::Trans || trans == Op:ConjTrans), m otherwise.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup trsm

template< typename Ta, typename Tb, typename Td = blas::scalar_type<Ta, Tb> >
void trsm(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    blas::scalar_type<Ta, Tb> const &alpha,
    Ta const *A, idx_int lda,
    Tb       *B, idx_int ldb,
    Td *w = nullptr);

#ifdef CBLAS_ROUTINES
template<>
void trsm<float, float>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    float const &alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb,
    float       *w);

template<>
void trsm<double, double>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    double const &alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb,
    double       *w);

template<>
void trsm<std::complex<float>, std::complex<float> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb,
    std::complex<float>       *w);

template<>
void trsm<std::complex<double>, std::complex<double> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb,
    std::complex<double>       *w);

#endif

}

#endif

