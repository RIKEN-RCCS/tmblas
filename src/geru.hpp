//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _GERU_HPP
#define _GERU_HPP

namespace tmblas {

// =============================================================================
/// General matrix rank-1 update:
/// \[
///     A = \alpha x y^H + A,
/// \]
/// where alpha is a scalar, x and y are vectors,
/// and A is an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] m
///     Number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not updated.
///
/// @param[in] x
///     The m-element vector x, in an array of length (m-1)*abs(incx) + 1.
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
///     The m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, m) [RowMajor: lda >= max(1, n)].
///
/// @ingroup geru

template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> > 
void geru(
  idx_int m, idx_int n,
  blas::scalar_type<Ta, Tb, Tc> const &alpha,
  Ta const *x, idx_int incx,
  Tb const *y, idx_int incy,
  Tc *A, idx_int lda); 

#ifdef CBLAS_ROUTINES
template<>
void geru<std::complex<float>, std::complex<float>, std::complex<float> >(
  idx_int m, idx_int n,
  std::complex<float> const &alpha,
  std::complex<float> const *x, idx_int incx,
  std::complex<float> const *y, idx_int incy,
  std::complex<float> *A, idx_int lda);

template<>
void geru<std::complex<double>, std::complex<double>, std::complex<double> >(
  idx_int m, idx_int n,
  std::complex<double> const &alpha,
  std::complex<double> const *x, idx_int incx,
  std::complex<double> const *y, idx_int incy,
  std::complex<double> *A, idx_int lda);
#endif

}

#endif
