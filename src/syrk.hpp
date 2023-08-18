//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SYRK_HPP
#define _SYRK_HPP

namespace tmblas {

// =============================================================================
/// Symmetric rank-k update:
/// \[
///     C = \alpha A A^T + \beta C,
/// \]
/// or
/// \[
///     C = \alpha A^T A + \beta C,
/// \]
/// where alpha and beta are scalars, C is an n-by-n symmetric matrix,
/// and A is an n-by-k or k-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] uplo
///     What part of the matrix C is referenced,
///     the opposite triangle being assumed from symmetry:
///     - Uplo::Lower: only the lower triangular part of C is referenced.
///     - Uplo::Upper: only the upper triangular part of C is referenced.
///
/// @param[in] trans
///     The operation to be performed:
///     - Op::NoTrans: $C = \alpha A A^T + \beta C$.
///     - Op::Trans:   $C = \alpha A^T A + \beta C$.
///     - In the real    case, Op::ConjTrans is interpreted as Op::Trans.
///       In the complex case, Op::ConjTrans is illegal (see @ref herk instead).
///
/// @param[in] n
///     Number of rows and columns of the matrix C. n >= 0.
///
/// @param[in] k
///     - If trans = NoTrans: number of columns of the matrix A. k >= 0.
///     - Otherwise:          number of rows    of the matrix A. k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not accessed.
///
/// @param[in] A
///     - If trans = NoTrans:
///       the n-by-k matrix A, stored in an lda-by-k array [RowMajor: n-by-lda].
///     - Otherwise:
///       the k-by-n matrix A, stored in an lda-by-n array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If trans = NoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)],
///     - Otherwise:              lda >= max(1, k) [RowMajor: lda >= max(1, n)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The n-by-n symmetric matrix C,
///     stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, n).
///
/// @param w
///     Work array of size n.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup syrk

template< typename Ta, typename Tb, typename Td = blas::scalar_type<Ta, Tb> >
void syrk(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    blas::scalar_type<Ta, Tb> const &alpha,
    Ta const *A, idx_int lda,
    blas::scalar_type<Ta, Tb> const &beta,
    Tb       *C, idx_int ldc,
    Td       *w = nullptr);

#ifdef CBLAS_ROUTINES
template<>
void syrk<float, float>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float const &alpha,
    float const *A, idx_int lda,
    float const &beta,
    float       *C, idx_int ldc,
    float       *w);

template<>
void syrk<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template<>
void syrk<double, double>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double const &alpha,
    double const *A, idx_int lda,
    double const &beta,
    double       *C, idx_int ldc,
    double       *w);

template<>
void syrk<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w);

#endif

}

#endif

