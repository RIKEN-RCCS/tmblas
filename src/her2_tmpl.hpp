//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _HER2_TMPL_HPP
#define _HER2_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Hermitian matrix rank-2 update:
/// \[
///     A = \alpha x y^H + \alpha y x^H + A,
/// \]
/// where alpha is a scalar, x and y are vectors,
/// and A is an n-by-n Hermitian matrix.
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
/// @ingroup her2

template< typename Ta, typename Tb, typename Tc, typename Td >
void her2(
    blas::Uplo  uplo,
    idx_int n,
    blas::scalar_type<Ta, Tb, Tc> const &alpha,
    Ta const *x, idx_int incx,
    Tb const *y, idx_int incy,
    Tc *A, idx_int lda)
{
  typedef blas::scalar_type<Ta, Tb, Tc> Tscalar;
  //  typedef blas::real_type<Tscalar> Tscalarreal;
  typedef blas::real_type<Td> Tdreal;
  typedef blas::real_type<Tc> Tcreal;
  
  #define A(i_, j_) A[ (i_) + (j_)*lda ]

  // constants
  //  const Tscalar Tscalar_zero(Tscalarreal(0));
  //  const Tdreal Tdreal_zero(0);
  //  const Tdreal Tdreal_two(2);

  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper );
  blas_error_if( n < 0 );
  blas_error_if( incx == 0 );
  blas_error_if( incy == 0 );
  blas_error_if( lda < n );

  // quick return
  if (n == 0 || mixedp_eq<Tscalar, int>(alpha, 0)) {
    return;
  }

  idx_int kx = (incx > 0 ? 0 : (-n + 1)*incx);
  idx_int ky = (incy > 0 ? 0 : (-n + 1)*incy);
  Td tmp1,tmp2,tmp3;
  if (uplo == blas::Uplo::Upper) {
    if (incx == 1 && incy == 1) {
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
        // note: NOT skipping if x[j] or y[j] is zero, for consistent NAN handling
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, conjg<Tb>(y[j]));
        mixedp_mul<Td, Tscalar, Ta>(tmp2, conjg(alpha), conjg<Ta>(x[j]));
        //scalar_t tmp1 = alpha * y[j];
        //scalar_t tmp2 = alpha * x[j];
        for (idx_int i = 0; i < j; ++i) {
          tmp3 = type_conv<Td, Tc>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp3, x[i], tmp1);
          mixedp_madd<Td, Tb, Td>(tmp3, y[i], tmp2);
          A(i, j) = type_conv<Tc, Td>(tmp3);
//          A(i, j) += x[i]*tmp1 + y[i]*tmp2;
        }
	{ // i == j
	  Tdreal tmp4;
	  tmp4 = type_conv<Tdreal, Tcreal>(real<Tcreal>(A(j, j)));
//  alpha x[j] conj(y[j]) + conj(alpha) conj(x[j]) y[j] =
//  2 Re (alpha x[j] conj(y[j]) is used in her2k  
//	  mixedp_madd<Td, Ta, Td>(tmp3, x[j], tmp1);
//	  mixedp_madd<Tdreal, Tdreal, Tdreal>(tmp4, real(tmp3), Tdreal_two);
	  mixedp_mul<Td, Ta, Td>(tmp3, x[j], tmp1);
	  mixedp_madd<Td, Tb, Td>(tmp3, y[j], tmp2);
	  mixedp_add<Tdreal, Tdreal, Tdreal>(tmp4, real<Tdreal>(tmp3), tmp4);
	  A(j, j) = type_conv<Tc, Tdreal>(tmp4);
	}
      }
    }
    else {
      // non-unit stride
      idx_int jx = kx;
      idx_int jy = ky;
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, conjg<Tb>(y[jy]));
        mixedp_mul<Td, Tscalar, Ta>(tmp2, conjg<Tscalar>(alpha), conjg<Ta>(x[jx]));
        //scalar_t tmp1 = alpha * y[jy];
        //scalar_t tmp2 = alpha * x[jx];
        idx_int ix = kx;
        idx_int iy = ky;
        for (idx_int i = 0; i < j; ++i) {
          tmp3 = type_conv<Td, Tc>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp3, x[ix], tmp1);
          mixedp_madd<Td, Tb, Td>(tmp3, y[iy], tmp2);
          A(i, j) = type_conv<Tc, Td>(tmp3);
          //A(i, j) += x[ix]*tmp1 + y[iy]*tmp2;
          ix += incx;
          iy += incy;
        }
	{ // i == j
	  Tdreal tmp4;
	  tmp4 = type_conv<Tdreal, Tcreal>(real<Tcreal>(A(j, j)));
//  alpha x[jx] conj(y[jy]) + conj(alpha) conj(x[jx]) y[jy] =
//  2 Re (alpha x[jx] conj(y[jy]) is used in her2k
//	  mixedp_madd<Td, Ta, Td>(tmp3, x[jx], tmp1);
//	  mixedp_madd<Tdreal, Tdreal, Tdreal>(tmp4, real(tmp3), Tdreal_two);
	  mixedp_mul<Td, Ta, Td>(tmp3, x[jx], tmp1);
	  mixedp_madd<Td, Tb, Td>(tmp3, y[jy], tmp2);
	  mixedp_add<Tdreal, Tdreal, Tdreal>(tmp4, real<Tdreal>(tmp3), tmp4);
	  A(j, j) = type_conv<Tc, Tdreal>(tmp4);
	  ix += incx;
          iy += incy;
	}
        jx += incx;
        jy += incy;
      }
    }
  }
  else {
    // lower triangle
    if (incx == 1 && incy == 1) {
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, conjg<Tb>(y[j]));
        mixedp_mul<Td, Tscalar, Ta>(tmp2, conjg<Tscalar>(alpha), conjg<Ta>(x[j]));
        //scalar_t tmp1 = alpha * y[j];
        //scalar_t tmp2 = alpha * x[j];
	{ // i == j
	  Tdreal tmp4;
	  tmp4 = type_conv<Tdreal, Tcreal>(real<Tcreal>(A(j, j)));
//  alpha x[j] conj(y[j]) + conj(alpha) conj(x[j]) y[j] =
//  2 Re (alpha x[j] conj(y[j]) is used in her2k
//	  mixedp_madd<Td, Ta, Td>(tmp3, x[j], tmp1);
//	  mixedp_madd<Tdreal, Tdreal, Tdreal>(tmp4, real(tmp3), Tdreal_two);
	  mixedp_mul<Td, Ta, Td>(tmp3, x[j], tmp1);
	  mixedp_madd<Td, Tb, Td>(tmp3, y[j], tmp2);
	  mixedp_add<Tdreal, Tdreal, Tdreal>(tmp4, real<Tdreal>(tmp3), tmp4);
	  A(j, j) = type_conv<Tc, Tdreal>(tmp4);
	}
        for (idx_int i = j + 1; i < n; ++i) {
          tmp3 = type_conv<Td, Tc>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp3, x[i], tmp1);
          mixedp_madd<Td, Tb, Td>(tmp3, y[i], tmp2);
          A(i, j) = type_conv<Tc, Td>(tmp3);
          //A(i, j) += x[i]*tmp1 + y[i]*tmp2;
        }
      }
    }
    else {
      // non-unit stride
      idx_int jx = kx;
      idx_int jy = ky;
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, conjg<Tb>(y[jy]));
        mixedp_mul<Td, Tscalar, Ta>(tmp2, conjg<Tscalar>(alpha), conjg<Ta>(x[jx]));
        //scalar_t tmp1 = alpha * y[jy];
        //scalar_t tmp2 = alpha * x[jx];
        idx_int ix = jx;
        idx_int iy = jy;
	{ // i == j 
	  Tdreal tmp4;
	  tmp4 = type_conv<Tdreal, Tcreal>(real<Tcreal>(A(j, j)));
//  alpha x[jx] conj(y[jy]) + conj(alpha) conj(x[jx]) y[jy] =
//  2 Re (alpha x[jx] conj(y[jy]) is used in her2k
//	  mixedp_madd<Td, Ta, Td>(tmp3, x[jx], tmp1);
//	  mixedp_madd<Tdreal, Tdreal, Tdreal>(tmp4, real(tmp3), Tdreal_two);
	  mixedp_mul<Td, Ta, Td>(tmp3, x[jx], tmp1);
	  mixedp_madd<Td, Tb, Td>(tmp3, y[jy], tmp2);
	  mixedp_add<Tdreal, Tdreal, Tdreal>(tmp4, real<Tdreal>(tmp3), tmp4);
	  A(j, j) = type_conv<Tc, Tdreal>(tmp4);
	  ix += incx;
	  iy += incy;
	}
        for (idx_int i = j + 1; i < n; ++i) {
          tmp3 = type_conv<Td, Tc>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp3, x[ix], tmp1);
          mixedp_madd<Td, Tb, Td>(tmp3, y[iy], tmp2);
          A(i, j) = type_conv<Tc, Td>(tmp3);
          //A(i, j) += x[ix]*tmp1 + y[iy]*tmp2;
          ix += incx;
          iy += incy;
        }
        jx += incx;
        jy += incy;
      }
    }
  }

  #undef A
}

}
#endif
