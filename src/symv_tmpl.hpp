//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SYMV_TMPL_HPP
#define _SYMV_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Symmetric matrix-vector multiply:
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
///     Work array of size n
///     Will be allocated/deallocated within the function if not given.
/// @ingroup symv

template<typename Ta, typename Tb, typename Tc, typename Td>
void symv(
  blas::Uplo uplo,
  idx_int n,
  blas::scalar_type<Ta, Tb, Tc> const &alpha,
  Ta const *A, idx_int lda,
  Tb const *x, idx_int incx,
  blas::scalar_type<Ta, Tb, Tc> const &beta,
  Tc *y, idx_int incy, Td *w)
{
  typedef blas::scalar_type<Ta, Tb, Tc> Tscalar;
  typedef blas::real_type<Tscalar> Tscalarreal;

  typedef blas::real_type<Td> Tdreal;  
  const Td Td_zero(Tdreal(0));
  bool hasworking = (!(w == (Td *)nullptr));
  Td *ww;

  #define A(i_, j_) A[ (i_) + (j_)*lda ]

  // constants
  const Tscalar Tscalar_zero(Tscalarreal(0)), Tscalar_one(Tscalarreal(1));
  
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper );
  blas_error_if( n < 0 );
  blas_error_if( lda < n );
  blas_error_if( incx == 0 );
  blas_error_if( incy == 0 );

  // quick return
  if (n == 0 || (alpha == Tscalar_zero && beta == Tscalar_one)) {
    return;
  }

  idx_int kx = (incx > 0 ? 0 : (-n + 1)*incx);
  idx_int ky = (incy > 0 ? 0 : (-n + 1)*incy);

  if (!hasworking) {
    ww = new Td[n];
  } 
  else {
    ww = w;
  }
  // form y = beta*y
  if (beta != Tscalar_one) {
    if (incy == 1) {
      if (beta == Tscalar_zero) {
        for (idx_int i = 0; i < n; ++i) {
          ww[i] = Td_zero;
        }
      }
      else {
        for (idx_int i = 0; i < n; ++i) {
          mixedp_mul<Td, Tc, Tscalar>(ww[i], y[i], beta);
          //y[i] *= beta;
        }
      }
    }
    else {
      idx_int iy = ky;
      if (beta == Tscalar_zero) {
        for (idx_int i = 0; i < n; ++i) {
          ww[i] = Td_zero;
        }
      }
      else {
        for (idx_int i = 0; i < n; ++i) {
          mixedp_mul<Td, Tc, Tscalar>(ww[i], y[iy], beta);
//          y[iy] *= beta;
          iy += incy;
        }
      }
    }
  } 
  else {
    int iy = ky;
    for(idx_int i=0 ; i< n ; ++i) {
      ww[i] = type_conv<Td, Tc>(y[iy]);
      iy += incy;
    }
  }
  if (alpha == Tscalar_zero) {
    if (incy == 1) {
      for (idx_int i = 0; i < n; ++i) {
	y[i] = type_conv<Tc, Td>(ww[i]);
      }
    }
    else {
      idx_int iy = ky;
      for (idx_int i = 0; i < n; ++i) {
	y[iy] = type_conv<Tc, Td>(ww[i]);
	iy += incy;
      }
    }
    return;
  }

  Td tmp1,tmp2;
  if (uplo == blas::Uplo::Upper) {
    // A is stored in upper triangle
    // form y += alpha * A * x
    if (incx == 1 && incy == 1) {
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
        //scalar_t tmp1 = alpha*x[j];
        //scalar_t tmp2 = zero;
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, x[j]);
        tmp2 = Td_zero;
        for (idx_int i = 0; i < j; ++i) {
          mixedp_madd<Td, Td, Ta>(ww[i], tmp1, A(i, j));
          mixedp_madd<Td, Ta, Tb>(tmp2, A(i, j), x[i]);
          //y[i] += tmp1 * A(i, j);
          //tmp2 += A(i, j) * x[i];
        }
        mixedp_madd<Td, Td, Ta>(ww[j], tmp1, A(j, j));
        mixedp_madd<Td, Tscalar, Td>(ww[j], alpha, tmp2);
        //y[j] += tmp1 * A(j, j) + alpha * tmp2;
      }
    }
    else {
      // non-unit stride
      idx_int jx = kx;
      for (idx_int j = 0; j < n; ++j) {
        //scalar_t tmp1 = alpha*x[jx];
        //scalar_t tmp2 = zero;
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, x[jx]);
        tmp2 = Td_zero;
        idx_int ix = kx;
        for (idx_int i = 0; i < j; ++i) {
          mixedp_madd<Td, Td, Ta>(ww[i], tmp1, A(i, j));
          mixedp_madd<Td, Ta, Tb>(tmp2, A(i, j), x[ix]);
//          y[iy] += tmp1 * A(i, j);
//          tmp2 += A(i, j) * x[ix];
          ix += incx;
        }
        mixedp_madd<Td, Td, Ta>(ww[j], tmp1, A(j, j));
        mixedp_madd<Td, Tscalar, Td>(ww[j], alpha, tmp2);

        //y[jy] += tmp1 * A(j, j) + alpha * tmp2;
        jx += incx;
      }
    }
  }
  else {
    // A is stored in lower triangle
    // form y += alpha * A * x
    if (incx == 1 && incy == 1) {
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
        //scalar_t tmp1 = alpha*x[j];
        //scalar_t tmp2 = zero;
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, x[j]);
        tmp2 = Td_zero;
        for (idx_int i = j+1; i < n; ++i) {
          mixedp_madd<Td, Td, Ta>(ww[i], tmp1, A(i, j));
          mixedp_madd<Td, Ta, Tb>(tmp2, A(i, j), x[i]);
          //y[i] += tmp1 * A(i, j);
          //tmp2 += A(i, j) * x[i];
        }
        mixedp_madd<Td, Td, Ta>(ww[j], tmp1, A(j, j));
        mixedp_madd<Td, Tscalar, Td>(ww[j], alpha, tmp2);
        //y[j] += tmp1 * A(j, j) + alpha * tmp2;
      }
    }
    else {
      // non-unit stride
      idx_int jx = kx;
      for (idx_int j = 0; j < n; ++j) {
        //scalar_t tmp1 = alpha*x[jx];
        //scalar_t tmp2 = zero;
        mixedp_mul<Td, Tscalar, Tb>(tmp1, alpha, x[jx]);
        tmp2 = Td_zero;
        idx_int ix = kx + incx * (j + 1) ;
        for (idx_int i = j+1; i < n; ++i) {
          mixedp_madd<Td, Td, Ta>(ww[i], tmp1, A(i, j));
          mixedp_madd<Td, Ta, Tb>(tmp2, A(i, j), x[ix]);
//          y[iy] += tmp1 * A(i, j);
//          tmp2 += A(i, j) * x[ix];
          ix += incx;
        }
        mixedp_madd<Td, Td, Ta>(ww[j], tmp1, A(j, j));
        mixedp_madd<Td, Tscalar, Td>(ww[j], alpha, tmp2);
        //y[jy] += tmp1 * A(j, j) + alpha * tmp2;
        jx += incx;
      }
    }
  }
  
  if (incy == 1) {
    for (idx_int i = 0; i < n; ++i) {
      y[i] = type_conv<Tc, Td>(ww[i]);
    }
  }
  else {
    idx_int iy = ky;
    for (idx_int i = 0; i < n; ++i) {
      y[iy] = type_conv<Tc, Td>(ww[i]);
      iy += incy;
    }
  }

  if(!hasworking) {
    delete [] ww;
  }
  
  #undef A

}

}

#endif
