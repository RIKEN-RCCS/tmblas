//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _HEMV_HPP
#define _HEMV_HPP

namespace tmblas {

// =============================================================================
/// Hermitian matrix-vector multiply:
/// \[
///     y = \alpha A x + \beta y,
/// \]
/// where alpha and beta are scalars, x and y are vectors,
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
///     Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, y need not be set on input.
///
/// @param[in, out] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
/// @param w
///     Work array of size n*incy.
///     Will be allocated/deallocated within the function if not given.
/// @ingroup hemv

template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> >
void hemv(
  blas::Uplo uplo,
  idx_int n,
  blas::scalar_type<Ta, Tb, Tc> const &alpha,
  Ta const *A, idx_int lda,
  Tb const *x, idx_int incx,
  blas::scalar_type<Ta, Tb, Tc> const &beta,
  Tc *y, idx_int incy, Td *w = nullptr);


#ifdef CBLAS_ROUTINES
template<>
void hemv<float, float, float>(
  blas::Uplo uplo,
  idx_int n,
  float const &alpha,
  float const *A, idx_int lda,
  float const *x, idx_int incx,
  float const &beta,
  float *y, idx_int incy, float *w);

template<>
void hemv<double, double, double>(
  blas::Uplo uplo,
  idx_int n,
  double const &alpha,
  double const *A, idx_int lda,
  double const *x, idx_int incx,
  double const &beta,
  double *y, idx_int incy, double *w);

#endif

}

#endif

