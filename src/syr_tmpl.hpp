//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SYR_TMPL_HPP
#define _SYR_TMPL_HPP

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
/// @ingroup syr

template<typename Ta, typename Tb, typename Td>
void syr(
    blas::Uplo uplo,
    idx_int n,
    blas::scalar_type<Ta, Tb> const &alpha,
    Ta const *x, idx_int incx,
    Tb       *A, idx_int lda)
{
  typedef blas::scalar_type<Ta, Tb> Tscalar;
  typedef blas::real_type<Tscalar> Tscalarreal;

  #define A(i_, j_) A[ (i_) + (j_)*lda ]

  // constants
  const Tscalar Tscalar_zero(Tscalarreal(0));
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper );
  blas_error_if( n < 0 );
  blas_error_if( incx == 0 );
  blas_error_if( lda < n );

  // quick return
  if (n == 0 || alpha == Tscalar_zero) {
    return;
  }

  idx_int kx = (incx > 0 ? 0 : (-n + 1)*incx);
  Td tmp, tmp2;
  if (uplo == blas::Uplo::Upper) {
    if (incx == 1) {
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
        // note: NOT skipping if x[j] is zero, for consistent NAN handling
        mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, x[j]);
        //scalar_t tmp = alpha * x[j];
        for (idx_int i = 0; i <= j; ++i) {
          tmp2 = type_conv<Td, Tb>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp2, x[i], tmp);
          A(i, j) = type_conv<Tb, Td>(tmp2);
          //A(i, j) += x[i] * tmp;
        }
      }
    }
    else {
      // non-unit stride
      idx_int jx = kx;
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, x[jx]);
        //scalar_t tmp = alpha * x[jx];
        idx_int ix = kx;
        for (idx_int i = 0; i <= j; ++i) {
          tmp2 = type_conv<Td, Tb>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp2, x[ix], tmp);
          A(i, j) = type_conv<Tb, Td>(tmp2);
          //A(i, j) += x[ix] * tmp;
          ix += incx;
        }
        jx += incx;
      }
    }
  }
  else {
    // lower triangle
    if (incx == 1) {
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, x[j]);
        //scalar_t tmp = alpha * x[j];
        for (idx_int i = j; i < n; ++i) {
          tmp2 = type_conv<Td, Tb>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp2, x[i], tmp);
          A(i, j) = type_conv<Tb, Td>(tmp2);
          //A(i, j) += x[i] * tmp;
        }
      }
    }
    else {
      // non-unit stride
      idx_int jx = kx;
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, x[jx]);
        //scalar_t tmp = alpha * x[jx];
        idx_int ix = jx;
        for (idx_int i = j; i < n; ++i) {
          tmp2 = type_conv<Td, Tb>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp2, x[ix], tmp);
          A(i, j) = type_conv<Tb, Td>(tmp2);
          //A(i, j) += x[ix] * tmp;
          ix += incx;
        }
        jx += incx;
      }
    }
  }

  #undef A
}

}
#endif
