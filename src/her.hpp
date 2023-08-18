//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _HER_HPP
#define _HER_HPP

namespace tmblas {

// =============================================================================
/// Symmetric matrix rank-1 update:
/// \[
///     A = \alpha x x^T + A,
/// \]
/// where alpha is a scalar, x is a vector,
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
/// @param[in, out] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @ingroup her

template<typename Ta, typename Tb, typename Td = blas::scalar_type<Ta, Tb> > 
void her(
    blas::Uplo uplo,
    idx_int n,
    blas::real_type<Ta, Tb> const &alpha,
    Ta const *x, idx_int incx,
    Tb       *A, idx_int lda);

#ifdef CBLAS_ROUTINES
template<>
void her<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template<>
void her<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double>       *A, idx_int lda);
#endif

}

#endif

