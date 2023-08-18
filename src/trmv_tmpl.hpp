//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TRMV_TMPL_HPP
#define _TRMV_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Triangular matrix-vector multiply:
/// \[
///     x = op(A) x,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// x is a vector,
/// and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero.
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///
/// @param[in] trans
///     The operation to be performed:
///     - Op::NoTrans:   $x = A   x$,
///     - Op::Trans:     $x = A^T x$,
///     - Op::ConjTrans: $x = A^H x$.
///
/// @param[in] diag
///     Whether A has a unit or non-unit diagonal:
///     - Diag::Unit:    A is assumed to be unit triangular.
///                      The diagonal elements of A are not referenced.
///     - Diag::NonUnit: A is not assumed to be unit triangular.
///
/// @param[in] n
///     Number of rows and columns of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @param[in, out] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param w
///     Work array of size n
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup trmv


template< typename Ta, typename Tb, typename Td >
void trmv(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    Ta const *A, idx_int lda,
    Tb       *x, idx_int incx,
    Td *w)
{
  bool hasworking = (!(w == (Td *)nullptr));
  Td *ww;
  
  #define A(i_, j_) A[ (i_) + (j_)*lda ]
  // check arguments
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper );
  blas_error_if( trans != blas::Op::NoTrans &&
                 trans != blas::Op::Trans &&
                 trans != blas::Op::ConjTrans );
  blas_error_if( diag != blas::Diag::NonUnit &&
                 diag != blas::Diag::Unit );
  blas_error_if( n < 0 );
  blas_error_if( lda < n );
  blas_error_if( incx == 0 );

  // quick return
  if (n == 0) {
    return;
  }

  bool nonunit = (diag == blas::Diag::NonUnit);
  idx_int kx = (incx > 0 ? 0 : (-n + 1)*incx);

  if (!hasworking) {
    ww = new Td[n];
  } 
  else {
    ww = w;
  }

  if (incx == 1) {
    for (idx_int i = 0; i < n; ++i) {
      ww[i] = type_conv<Td, Tb>(x[i]);
    }
  }
  else {
    idx_int ix = kx;
    for (idx_int i = 0; i < n; ++i) {
      ww[i] = type_conv<Td, Tb>(x[ix]);
      ix += incx;
    }
  }
  
  Td tmp;
  if (trans == blas::Op::NoTrans) {
    // Form x := A*x
    if (uplo == blas::Uplo::Upper) {
      // upper
      for (idx_int j = 0; j < n; ++j) {
	// note: NOT skipping if x[j] is zero, for consistent NAN handling
	//TX tmp = x[j];
	tmp = ww[j];
	for (idx_int i = 0; i <= j-1; ++i) {
	  mixedp_madd<Td, Td, Ta>(ww[i], tmp, A(i, j));
	  //x[i] += tmp * A(i, j);
	}
	if (nonunit) {
	  mixedp_mul<Td, Td, Ta>(ww[j], ww[j], A(j, j));
	  //x[j] *= A(j, j);
	}
      }
    }
    else {
      // lower
      // unit stride
      for (idx_int j = n-1; j >= 0; --j) {
	// note: NOT skipping if x[j] is zero ...
	//TX tmp = x[j];
	tmp = ww[j];
	for (idx_int i = n-1; i >= j+1; --i) {
	  mixedp_madd<Td, Td, Ta>(ww[i], tmp, A(i, j));
	  //x[i] += tmp * A(i, j);
	}
	if (nonunit) {
	  mixedp_mul<Td, Td, Ta>(ww[j], ww[j], A(j, j));
	  //x[j] *= A(j, j);
	}
      }
    }
  }
  else if (trans == blas::Op::Trans) {
    // Form  x := A^T * x
    if (uplo == blas::Uplo::Upper) {
      // upper
      // unit stride
      for (idx_int j = n-1; j >= 0; --j) {
	//TX tmp = x[j];
	tmp = ww[j];
	if (nonunit) {
	  //tmp *= A(j, j);
	  mixedp_mul<Td, Td, Ta>(tmp, tmp, A(j, j));
	}
	for (idx_int i = j - 1; i >= 0; --i) {
	  //tmp += A(i, j) * x[i];
	  mixedp_madd<Td, Ta, Td>(tmp, A(i, j), ww[i]);
	}
	//x[j] = tmp;
	ww[j] = tmp;
      }
    }
    else {
      // lower
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
	//TX tmp = x[j];
	tmp = ww[j];
	if (nonunit) {
	  //tmp *= A(j, j);
	  mixedp_mul<Td, Td, Ta>(tmp, tmp, A(j, j));
	}
	for (idx_int i = j + 1; i < n; ++i) {
	  //tmp += A(i, j) * x[i];
	  mixedp_madd<Td, Ta, Td>(tmp, A(i, j), ww[i]);
	}
	//x[j] = tmp;
	ww[j] = tmp;
      }
    }
  }
  else {
    // Form x := A^H * x
    // same code as above A^T * x case, except add conj()
    if (uplo == blas::Uplo::Upper) {
      // upper
      // unit stride
      for (idx_int j = n-1; j >= 0; --j) {
	//TX tmp = x[j];
	tmp = ww[j];
	if (nonunit) {
	  //tmp *= conj( A(j, j) );
	  mixedp_mul<Td, Td, Ta>(tmp, tmp, conjg<Ta>(A(j, j)));
	}
	for (idx_int i = j - 1; i >= 0; --i) {
	  //tmp += conj( A(i, j) ) * x[i];
	  mixedp_madd<Td, Ta, Td>(tmp, conjg<Ta>(A(i, j)), ww[i]);
	}
	//x[j] = tmp;
	ww[j] = tmp;
      }
    }
    else {
      // lower
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
	//TX tmp = x[j];
	tmp = ww[j];
	if (nonunit) {
	  //tmp *= conj( A(j, j) );
	  mixedp_mul<Td, Td, Ta>(tmp, tmp, conjg<Ta>(A(j, j)));
	}
	for (idx_int i = j + 1; i < n; ++i) {
	  //tmp += conj( A(i, j) ) * x[i];
	  mixedp_madd<Td, Ta, Td>(tmp, conjg<Ta>(A(i, j)), ww[i]);
	}
	//x[j] = tmp;
	ww[j] = tmp;
      }
    }
  }
  if (incx == 1) {
    for (idx_int i = 0; i < n; ++i) {
      x[i] = type_conv<Tb, Td>(ww[i]);
    }
  }
  else {
    idx_int ix = kx;
    for (idx_int i = 0; i < n; ++i) {
      x[ix] = type_conv<Tb, Td>(ww[i]);
      ix += incx;
    }
  }
  if (!hasworking) {
    delete [] ww;
  }

  #undef A
}

}
#endif
