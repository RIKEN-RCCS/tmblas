//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _GEMV_HPP
# define _GEMV_HPP

namespace tmblas{

// =============================================================================
/// General matrix-vector multiply:
/// \[
///     y = \alpha op(A) x + \beta y,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// alpha and beta are scalars, x and y are vectors,
/// and A is an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] trans
///     The operation to be performed:
///     - Op::NoTrans:   $y = \alpha A   x + \beta y$,
///     - Op::Trans:     $y = \alpha A^T x + \beta y$,
///     - Op::ConjTrans: $y = \alpha A^H x + \beta y$.
///
/// @param[in] m
///     Number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, m) [RowMajor: lda >= max(1, n)].
///
/// @param[in] x
///     - If trans = NoTrans:
///       the n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///     - Otherwise:
///       the m-element vector x, in an array of length (m-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, y need not be set on input.
///
/// @param[in, out] y
///     - If trans = NoTrans:
///       the m-element vector y, in an array of length (m-1)*abs(incy) + 1.
///     - Otherwise:
///       the n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @param w
///     Work array of size leny where leny = m if trans == Op::NotTrans, and m otherwise.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup gemv

template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> > 
void gemv(
     blas::Op trans,
     idx_int m, idx_int n,
     blas::scalar_type<Ta, Tb, Tc> const &alpha,
     Ta const *A, idx_int lda,
     Tb const *x, idx_int incx,
     blas::scalar_type<Ta, Tb, Tc> const &beta,
     Tc *y, idx_int incy);

#ifdef CBLAS_ROUTINES
template<>
void gemv<float, float, float>(
     blas::Op trans,
     idx_int m, idx_int n,
     float const &alpha,
     float const *A, idx_int lda,
     float const *x, idx_int incx,
     float const &beta,
     float *y, idx_int incy);

template<>
void gemv<double, double, double>(
     blas::Op trans,
     idx_int m, idx_int n,
     double const &alpha,
     double const *A, idx_int lda,
     double const *x, idx_int incx,
     double const &beta,
     double *y, idx_int incy);

template<>
void gemv<std::complex<float>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<float> const &alpha,
     std::complex<float> const *A, idx_int lda,
     std::complex<float> const *x, idx_int incx,
     std::complex<float> const &beta,
     std::complex<float> *y, idx_int incy);

template<>
void gemv<std::complex<double>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
     idx_int m, idx_int n,
     std::complex<double> const &alpha,
     std::complex<double> const *A, idx_int lda,
     std::complex<double> const *x, idx_int incx,
     std::complex<double> const &beta,
     std::complex<double> *y, idx_int incy);
#endif

}

#endif
