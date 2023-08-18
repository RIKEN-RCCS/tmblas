//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _GEMV_TMPL_HPP
# define _GEMV_TMPL_HPP

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
/// @ingroup gemv

//   only for  (layout == Layout::ColMajor)
template<typename Ta, typename Tb, typename Tc, typename Td>
void gemv(
     blas::Op trans,
     idx_int m, idx_int n,
     blas::scalar_type<Ta, Tb, Tc> const &alpha,
     Ta const *A, idx_int lda,
     Tb const *x, idx_int incx,
     blas::scalar_type<Ta, Tb, Tc> const &beta,
     Tc *y, idx_int incy)
{

  typedef blas::scalar_type<Ta, Tb, Tc> scalar_t;
  //  typedef blas::real_type<scalar_t> scalar_treal;  
  typedef blas::real_type<Td> Tdreal;
  typedef blas::real_type<Tc> Tcreal;  

#define A(i_, j_) A[ (i_) + (j_)*lda ]

  // constants
  //  const scalar_t zero(scalar_treal(0));
  //  const scalar_t one(scalar_treal(1));
  const Td Tdzero(Tdreal(0));
  const Tc Tczero(Tcreal(0));  

  // check arguments
  blas_error_if( trans != blas::Op::NoTrans &&
                 trans != blas::Op::Trans &&
                 trans != blas::Op::ConjTrans );
  blas_error_if( m < 0 );
  blas_error_if( n < 0 );
  
  blas_error_if( lda < m );
  
  blas_error_if( incx == 0 );
  blas_error_if( incy == 0 );
  
  // quick return
  if (m == 0 || n == 0 || (mixedp_eq<scalar_t, int>(alpha, 0) &&
			   mixedp_eq<scalar_t, int>(beta, 1)))
    return;

  idx_int lenx = (trans == blas::Op::NoTrans ? n : m);
  idx_int leny = (trans == blas::Op::NoTrans ? m : n);
  idx_int kx = (incx > 0 ? 0 : (-lenx + 1)*incx);
  idx_int ky = (incy > 0 ? 0 : (-leny + 1)*incy);

    // ----------
    // form y = beta*y
  if (mixedp_eq<scalar_t, int>(alpha, 0)) {
    if (!mixedp_eq<scalar_t, int>(beta, 1)) {
      if (incy == 1) {
	if (mixedp_eq<scalar_t, int>(beta, 0)) {
	  for (idx_int i = 0; i < leny; ++i) {
	    y[i] = Tczero;
	  }
	}
	else {
	  Td tmp;
	  for(idx_int i = 0 ; i < leny ; ++i) {
	    mixedp_mul<Td, Tc, scalar_t>(tmp, y[i], beta);
	    y[i] = type_conv<Tc, Td>(tmp);
	  }
	}
      } // if (incy == 1)
      else {
	idx_int iy = ky;
	if (mixedp_eq<scalar_t, int>(beta, 0)) {
	  for (idx_int i = 0; i < leny; ++i) {
	    y[iy] = Tczero;
	    iy += incy;
	  }
	}
	else {
	  Td tmp;
	  for(idx_int i = 0 ; i < leny ; ++i) {
	    mixedp_mul<Td, Tc, scalar_t>(tmp, y[iy], beta);
	    y[iy] = type_conv<Tc, Td>(tmp);
	    iy += incy;	    
	  }
	}
      } // if (incy == 1)(
    } // if (beta ! = one)
    return;
  }

  if (trans == blas::Op::NoTrans) {
    idx_int iy = ky;
    if (incx == 1) {
      for (idx_int i = 0; i < m; ++i) {
        Td tmp = Tdzero;
        for (idx_int j = 0; j < n; ++j) {
          //tmp += A(i, j) * x[j];
          mixedp_madd<Td, Ta, Tb>(tmp, A(i, j), x[j]);
        }
        //y[iy] = alpha*tmp + beta * y[iy];
        mixedp_mul<Td, scalar_t, Td>(tmp, alpha, tmp);
        mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[iy]);
        y[iy] = type_conv<Tc, Td>(tmp);
        iy += incy;
      }
    } else {
      for (idx_int i = 0; i < m; ++i) {
        Td tmp = Tdzero;
        idx_int jx = kx;
        for (idx_int j = 0; j < n; ++j) {
          //tmp += A(i, j) * x[jy];
          mixedp_madd<Td, Ta, Tb>(tmp, A(i, j), x[jx]);
          jx += incx;
        }
        //y[iy] += alpha*tmp;
        mixedp_mul<Td, scalar_t, Td>(tmp, alpha, tmp);
        mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[iy]);
        y[iy] = type_conv<Tc, Td>(tmp);
        iy += incy;
      }
    }
  }
  else if (trans == blas::Op::Trans) {
    idx_int jy = ky;
    if (incx == 1) {
      for (idx_int j = 0; j < n; ++j) {
        Td tmp = Tdzero;
        for (idx_int i = 0; i < m; ++i) {
          //tmp += A(i, j) * x[i];
          mixedp_madd<Td, Ta, Tb>(tmp, A(i, j), x[i]);
        }
        //y[jy] += alpha*tmp;
        mixedp_mul<Td, scalar_t, Td>(tmp, alpha, tmp);
        mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[jy]);
        y[jy] = type_conv<Tc, Td>(tmp);
        jy += incy;
      }
    } else {
      for (idx_int j = 0; j < n; ++j) {
        Td tmp = Tdzero;
        idx_int ix = kx;
        for (idx_int i = 0; i < m; ++i) {
          //tmp += A(i, j) * x[i];
          mixedp_madd<Td, Ta, Tb>(tmp, A(i, j), x[ix]);
          ix += incx;
        }
        //y[jy] += alpha*tmp;
        mixedp_mul<Td, scalar_t, Td>(tmp, alpha, tmp);
        mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[jy]);
        y[jy] = type_conv<Tc, Td>(tmp);
        jy += incy;
      }
    }
  }
  else { // if (trans == blas::Op::ConjTrans) {
    idx_int jy = ky;
    if (incx == 1) {
      for (idx_int j = 0; j < n; ++j) {
        Td tmp = Tdzero;
        for (idx_int i = 0; i < m; ++i) {
          //tmp += A(i, j) * x[i];
          mixedp_madd<Td, Ta, Tb>(tmp, conjg<Ta>(A(i, j)), x[i]);
        }
        //y[jy] += alpha*tmp;
        mixedp_mul<Td, scalar_t, Td>(tmp, alpha, tmp);
        mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[jy]);
        y[jy] = type_conv<Tc, Td>(tmp);
        jy += incy;
      }
    } else {
      for (idx_int j = 0; j < n; ++j) {
        Td tmp = Tdzero;
        idx_int ix = kx;
        for (idx_int i = 0; i < m; ++i) {
          //tmp += A(i, j) * x[i];
          mixedp_madd<Td, Ta, Tb>(tmp, conjg<Ta>(A(i, j)), x[ix]);
          ix += incx;
        }
        //y[jy] += alpha*tmp;
        mixedp_mul<Td, scalar_t, Td>(tmp, alpha, tmp);
        mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[jy]);
        y[jy] = type_conv<Tc, Td>(tmp);
        jy += incy;
      }
    }
  }

  #undef A
}
}
#endif

