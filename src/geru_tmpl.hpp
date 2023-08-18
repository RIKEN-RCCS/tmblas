//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _GERU_TMPL_HPP
#define _GERU_TMPL_HPP

namespace tmblas {

// =============================================================================
/// General complex matrix rank-1 update:
/// \[
///     A = \alpha x y^T + A,
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

template<typename Ta, typename Tb, typename Tc, typename Td>
void geru(
  idx_int m, idx_int n,
  blas::scalar_type<Ta, Tb, Tc> const &alpha,
  Ta const *x, idx_int incx,
  Tb const *y, idx_int incy,
  Tc *A, idx_int lda)
{
  typedef blas::scalar_type<Ta, Tb, Tc> scalar_t;
  typedef blas::real_type<scalar_t> scalar_treal;  

  #define A(i_, j_) A[ (i_) + (j_)*lda ]

  // constants
  const scalar_t zero(scalar_treal(0));

  // check arguments

  blas_error_if( m < 0 );
  blas_error_if( n < 0 );
  blas_error_if( incx == 0 );
  blas_error_if( incy == 0 );
  //     if (layout == Layout::ColMajor)
  blas_error_if( lda < m );
	
  // quick return
  if (m == 0 || n == 0 || alpha == zero) {
      return;
  }

  {
    if (incx == 1 && incy == 1) {
      // unit stride
      Td tmp;
      for (idx_int j = 0; j < n; ++j) {
        // note: NOT skipping if y[j] is zero, for consistent NAN handling
        //scalar_t tmp = alpha * conj( y[j] );
        mixedp_mul<Td, scalar_t, Tb>(tmp, alpha, y[j]);
        for (idx_int i = 0; i < m; ++i) {
          //A(i, j) += x[i] * tmp;
          Td tmp2 = type_conv<Td, Tc>(A(i, j));
          mixedp_madd<Td, Td, Ta>(tmp2, tmp, x[i]);
          A(i, j) = type_conv<Tc, Td>(tmp2);
        }
      }
    }
    else if (incx == 1) {
      // x unit stride, y non-unit stride
      idx_int jy = (incy > 0 ? 0 : (-n + 1)*incy);
      Td tmp;
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, scalar_t, Tb>(tmp, alpha, y[jy]);
        for (idx_int i = 0; i < m; ++i) {
          Td tmp2 = type_conv<Td, Tc>(A(i, j));
          mixedp_madd<Td, Td, Ta>(tmp2, tmp, x[i]);
          A(i, j) = type_conv<Tc, Td>(tmp2);
        }
        jy += incy;
      }
    }
    else {
      // x and y non-unit stride
      Td tmp;
      idx_int kx = (incx > 0 ? 0 : (-m + 1)*incx);
      idx_int jy = (incy > 0 ? 0 : (-n + 1)*incy);
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, scalar_t, Tb>(tmp, alpha, y[jy]);
        idx_int ix = kx;
        for (idx_int i = 0; i < m; ++i) {
          Td tmp2 = type_conv<Td, Tc>(A(i, j));
          mixedp_madd<Td, Td, Ta>(tmp2, tmp, x[ix]);
          A(i, j) = type_conv<Tc, Td>(tmp2);
          ix += incx;
        }
        jy += incy;
      }
    }
  }

  #undef A
}

}

#endif
