//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SYR2_HPP
#define _SYR2_HPP

namespace tmblas {

// =============================================================================
/// Symmetric matrix rank-2 update:
/// \[
///     A = \alpha x y^T + \alpha y x^T + A,
/// \]
/// where alpha is a scalar, x and y are vectors,
/// and A is an n-by-n symmetric matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed from symmetry.
///     - Uplo::Lower: only the lower triangular part of A is referenced.
///     - Uplo::Upper: only the upper triangular part of A is referenced.
///
/// @param[in] n
///     Number of rows and columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not updated.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @param[in, out] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @ingroup syr2

template< typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> >
void syr2(
    blas::Uplo  uplo,
    idx_int n,
    blas::scalar_type<Ta, Tb, Tc> const &alpha,
    Ta const *x, idx_int incx,
    Tb const *y, idx_int incy,
    Tc *A, idx_int lda);

#ifdef CBLAS_ROUTINES
template<>
void syr2<float, float, float>(
    blas::Uplo  uplo,
    idx_int n,
    float const & alpha,
    float const *x, idx_int incx,
    float const *y, idx_int incy,
    float *A, idx_int lda);

template<>
void syr2<double, double, double>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double const *y, idx_int incy,
    double *A, idx_int lda);
#endif

}

#endif

