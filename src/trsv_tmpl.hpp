//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TRSV_TMPL_HPP
#define _TRSV_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Solve the triangular matrix-vector equation
/// \[
///     op(A) x = b,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// x and b are vectors,
/// and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// @see LAPACK's latrs for a more numerically robust implementation.
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
///     The equation to be solved:
///     - Op::NoTrans:   $A   x = b$,
///     - Op::Trans:     $A^T x = b$,
///     - Op::ConjTrans: $A^H x = b$.
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
///     Work array of size n.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup trsv

template<typename Ta, typename Tb, typename Td> 
void
trsv(blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  Ta const *A, const idx_int lda,
	  Tb *X, const idx_int incx,
	  Td *w)
{
  idx_int kx;
  bool hasworking = (!(w == (Td *)nullptr));
  Td *ww;

  #define A(i_, j_) A[ (i_) + (j_)*lda ]

  // check arguments
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper );
  blas_error_if( transA != blas::Op::NoTrans &&
                 transA != blas::Op::Trans &&
                 transA != blas::Op::ConjTrans );
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
  kx = (incx > 0 ? 0 : (-n + 1)*incx);

  if (!hasworking) {
    ww = new Td[n];
  } // ! hasworking
  else {
    ww = w;
  }
  if (incx == 1) { 
    for (idx_int i=0 ; i<n ; ++i) {
      ww[i] = type_conv<Td, Tb>(X[i]);
    }
  }
  else {
    idx_int ix = kx;
    for (idx_int i= 0 ; i < n ; ++i) {
      ww[i] = type_conv<Td, Tb>(X[ix]);
      ix += incx;   // decrement with stride abs(incx) : due to incx < 0
    }
  }
  if (transA == blas::Op::NoTrans) {
    // Form x := A^{-1} * x
    if (uplo == blas::Uplo::Upper) {
      // upper
      for (idx_int j = n - 1; j >= 0; --j) {
	// note: NOT skipping if x[j] is zero, for consistent NAN handling
	if (nonunit) {
	  mixedp_div<Td, Td, Ta>(ww[j], ww[j], A(j,j));
	}
	Td temp = ww[j];
	for (idx_int i = j - 1; i >= 0; --i) {
	  mixedp_msub<Td,Td,Ta>(ww[i], temp, A(i, j));
	}
      }
    }
    else {
      // lower
      for (idx_int j = 0 ; j <n ; ++j) {
	// note: NOT skipping if x[j] is zero, for consistent NAN handling
	if (nonunit) {
	  mixedp_div<Td, Td, Ta>(ww[j], ww[j], A(j,j));
	}
	Td temp = ww[j];
	for (idx_int i = j + 1; i < n; ++i) {
	  mixedp_msub<Td,Td,Ta>(ww[i], temp, A(i, j));
	}
      }
    }
  } else if (transA == blas::Op::Trans) {
    // Form x := A^{-1} * x
    if (uplo == blas::Uplo::Upper) {
      // upper
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
	Td temp = ww[j];
	for (idx_int i = 0; i <= j-1; ++i) {
	  mixedp_msub<Td,Td,Ta>(temp, ww[i], A(i, j));
	}
	// note: NOT skipping if x[j] is zero, for consistent NAN handling
	if (nonunit) {
	  mixedp_div<Td, Td, Ta>(temp, temp, A(j,j));
	}
	ww[j] = temp;
      }
    }
    else {
      // lower
      for (idx_int j = n-1 ; j >= 0 ; --j) {
	// note: NOT skipping if x[j] is zero, for consistent NAN handling
	Td temp = ww[j];
	for (idx_int i = j + 1; i < n; ++i) {
	  mixedp_msub<Td,Td,Ta>(temp, ww[i], A(i, j));
	}
	if (nonunit) {
	  mixedp_div<Td, Td, Ta>(temp, temp, A(j,j));
	}
	ww[j] = temp;
      }
    }
  } 
  else { //ConjTrans
    // Form x := A^{-1} * x
    if (uplo == blas::Uplo::Upper) {
      // upper
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
	Td temp = ww[j];
	for (idx_int i = 0; i <= j-1; ++i) {
	  mixedp_msub<Td,Td,Ta>(temp, ww[i], conjg<Ta>(A(i, j)));
	}
	// note: NOT skipping if x[j] is zero, for consistent NAN handling
	if (nonunit) {
	  mixedp_div<Td, Td, Ta>(temp, temp, conjg<Ta>(A(j,j)));
	}
	ww[j] = temp;
      }
    }
    else {
      // lower
      for (idx_int j = n-1 ; j >= 0 ; --j) {
	// note: NOT skipping if x[j] is zero, for consistent NAN handling
	Td temp = ww[j];
	for (idx_int i = j + 1; i < n; ++i) {
	  mixedp_msub<Td,Td,Ta>(temp, ww[i], conjg<Ta>(A(i, j)));
	}
	if (nonunit) {
	  mixedp_div<Td, Td, Ta>(temp, temp, conjg<Ta>(A(j,j)));
	}
	ww[j] = temp;
      }
    }
  }

  if (incx == 1) { 
    for (idx_int i=0 ; i<n ; ++i) {
      X[i] = type_conv<Tb, Td>( ww[i] );
    }
  }
  else {
    idx_int ix = kx;
    for (idx_int i= 0 ; i < n ; ++i) {
      X[ix] = type_conv<Tb, Td>( ww[i] );
      ix += incx;   // decrement with stride abs(incx) : due to incx < 0
    }
  }

  if(!hasworking) {
    delete [] ww;
  }

  #undef A
}

}
#endif
