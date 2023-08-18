//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HER2_HH
#define BLAS_HER2_HH

#include "blas/util.hh"
#include "blas/syr2.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Hermitian matrix rank-2 update:
/// \[
///     A = \alpha x y^H + \text{conj}(\alpha) y x^H + A,
/// \]
/// where alpha is a scalar, x and y are vectors,
/// and A is an n-by-n Hermitian matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
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
///     Imaginary parts of the diagonal elements need not be set,
///     are assumed to be zero on entry, and are set to zero on exit.
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @ingroup her2

template <typename TA, typename TX, typename TY>
void her2(
    blas::Layout layout,
    blas::Uplo  uplo,
    int64_t n,
    blas::scalar_type<TA, TX, TY> alpha,
    TX const *x, int64_t incx,
    TY const *y, int64_t incy,
    TA *A, int64_t lda )
{
    typedef blas::scalar_type<TA, TX, TY> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero(0); //  29 Jun.2023 AS

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );
    blas_error_if( lda < n );

    // quick return
    if (n == 0 || alpha == zero)
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);
    int64_t ky = (incy > 0 ? 0 : (-n + 1)*incy);
    if (uplo == Uplo::Upper) {
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] or y[j] is zero, for consistent NAN handling
                scalar_t tmp1 = alpha * conj( y[j] );
                scalar_t tmp2 = conj( alpha * x[j] );
                for (int64_t i = 0; i <= j-1; ++i) {
                    A(i, j) += x[i]*tmp1 + y[i]*tmp2;
                }
                A(j, j) = real( A(j, j) ) + real( x[j]*tmp1 + y[j]*tmp2 );
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            int64_t jy = ky;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp1 = alpha * conj( y[jy] );
                scalar_t tmp2 = conj( alpha * x[jx] );
                int64_t ix = kx;
                int64_t iy = ky;
                for (int64_t i = 0; i <= j-1; ++i) {
                    A(i, j) += x[ix]*tmp1 + y[iy]*tmp2;
                    ix += incx;
                    iy += incy;
                }
                A(j, j) = real( A(j, j) ) + real( x[jx]*tmp1 + y[jy]*tmp2 );
                jx += incx;
                jy += incy;
            }
        }
    }
    else {
        // lower triangle
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp1 = alpha * conj( y[j] );
                scalar_t tmp2 = conj( alpha * x[j] );
                A(j, j) = real( A(j, j) ) + real( x[j]*tmp1 + y[j]*tmp2 );
                for (int64_t i = j+1; i < n; ++i) {
                    A(i, j) += x[i]*tmp1 + y[i]*tmp2;
                }
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            int64_t jy = ky;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp1 = alpha * conj( y[jy] );
                scalar_t tmp2 = conj( alpha * x[jx] );
                A(j, j) = real( A(j, j) ) + real( x[jx]*tmp1 + y[jy]*tmp2 );
                int64_t ix = jx;
                int64_t iy = jy;
                for (int64_t i = j+1; i < n; ++i) {
                    ix += incx;
                    iy += incy;
                    A(i, j) += x[ix]*tmp1 + y[iy]*tmp2;
                }
                jx += incx;
                jy += incy;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_HER2_HH
