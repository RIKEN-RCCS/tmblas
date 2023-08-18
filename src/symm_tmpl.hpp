//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SYMM_TMPL_HPP
#define _SYMM_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Symmetric matrix-matrix multiply:
/// \[
///     C = \alpha A B + \beta C,
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C,
/// \]
/// where alpha and beta are scalars, A is an m-by-m or n-by-n symmetric matrix,
/// and B and C are m-by-n matrices.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] side
///     The side the matrix A appears on:
///     - Side::Left:  $C = \alpha A B + \beta C$,
///     - Side::Right: $C = \alpha B A + \beta C$.
///
/// @param[in] uplo
///     What part of the matrix A is referenced:
///     - Uplo::Lower: only the lower triangular part of A is referenced.
///     - Uplo::Upper: only the upper triangular part of A is referenced.
///
/// @param[in] m
///     Number of rows of the matrices B and C.
///
/// @param[in] n
///     Number of columns of the matrices B and C.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If side = Left:  The m-by-m matrix A, stored in an lda-by-m array.
///     - If side = Right: The n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of A.
///     - If side = Left:  lda >= max(1, m).
///     - If side = Right: lda >= max(1, n).
///
/// @param[in] B
///     The m-by-n matrix B, stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, n).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C, stored in an lda-by-n array.
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, n).
///
/// @ingroup symm

template<typename Ta, typename Tb, typename Tc, typename Td >
void symm(
  blas::Side side,
  blas::Uplo uplo,
  idx_int m, idx_int n,
  blas::scalar_type<Ta, Tb, Tc> const &alpha,
  Ta const *A, idx_int lda,
  Tb const *B, idx_int ldb,
  blas::scalar_type<Ta, Tb, Tc> const &beta,
  Tc *C, idx_int ldc) 
{

  typedef blas::scalar_type<Ta, Tb, Tc> Tscalar;
  typedef blas::real_type<Tscalar> Tscalarreal;
  typedef blas::real_type<Tc> Tcreal;
  typedef blas::real_type<Td> Tdreal;  
  const Tscalar Tscalar_zero(Tscalarreal(0)), Tscalar_one(Tscalarreal(1));
  const Tc Tc_zero(Tcreal(0));
  const Td Td_zero(Tdreal(0));

  #define A(i_, j_) A[ (i_) + (j_)*lda ]
  #define B(i_, j_) B[ (i_) + (j_)*ldb ]
  #define C(i_, j_) C[ (i_) + (j_)*ldc ]

  blas_error_if( side != blas::Side::Left &&
                 side != blas::Side::Right );
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper &&
                 uplo != blas::Uplo::General );
  blas_error_if( m < 0 );
  blas_error_if( n < 0 );

  // check remaining arguments
  blas_error_if( lda < ((side == blas::Side::Left) ? m : n) );
  blas_error_if( ldb < m );
  blas_error_if( ldc < m );

  // quick return
  if (m == 0 || n == 0) {
    return;
  }

  // alpha == zero
  if (alpha == Tscalar_zero) {
    if (beta == Tscalar_zero) {
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i = 0; i < m; ++i) {
          C(i, j) = Tc_zero;
        }
      }
    }
    else if (beta != Tscalar_one) {
      Td tmp;
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i = 0; i < m; ++i) {
          mixedp_mul<Td, Tc, Tscalar>(tmp, C(i,j), beta);
          C(i, j) = type_conv<Tc, Td>(tmp);
        }
      }
    }
    return;
  }

    // alpha != zero
  if (side == blas::Side::Left) {
    if (uplo != blas::Uplo::Lower) {
    // uplo == Uplo::Upper or uplo == Uplo::General
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i = 0; i < m; ++i) {
          Td w;
          mixedp_mul<Td, Tc, Tscalar>(w, C(i, j), beta);
          Td s = Td_zero;
          for (idx_int k = 0; k <= i; ++k) {
            mixedp_madd<Td, Ta, Tb>(s, A(k, i), B(k, j));
          }
          for (idx_int k=i+1 ; k < m ; ++k) {
            mixedp_madd<Td, Ta, Tb>(s, A(i, k), B(k, j));
          }
          mixedp_madd<Td, Tscalar, Td>(w, alpha, s);
          C(i, j) = type_conv<Tc, Td>(w);
        }
      }
    }
    else {
    // uplo == Uplo::Lower
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i = 0; i < m; ++i) {
          Td w;
          mixedp_mul<Td, Tc, Tscalar>(w, C(i, j), beta);
          Td s = Td_zero;
          for (idx_int k = 0; k <= i; ++k) {
            mixedp_madd<Td, Ta, Tb>(s, A(i, k), B(k, j));
          }
          for (idx_int k=i+1 ; k < m ; ++k) {
            mixedp_madd<Td, Ta, Tb>(s, A(k, i), B(k, j));
          }
          mixedp_madd<Td, Tscalar, Td>(w, alpha, s);
          C(i, j) = type_conv<Tc, Td>(w);
        }
      }
    }
  }
  else {
    if (uplo != blas::Uplo::Lower) {
    // uplo == Uplo::Upper or uplo == Uplo::General
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i=0 ; i<m ; ++i) {
          Td w;
          mixedp_mul<Td, Tc, Tscalar>(w, C(i, j), beta);
          Td s = Td_zero;
          for (idx_int k = 0; k <= j; ++k) {
            mixedp_madd<Td, Tb, Ta>(s, B(i, k), A(k, j));
          }
          for (idx_int k = j+1 ; k<n ; ++k) {
            mixedp_madd<Td, Tb, Ta>(s, B(i, k), A(j, k));
          }
          mixedp_madd<Td, Tscalar, Td>(w, alpha, s);
          C(i, j) = type_conv<Tc, Td>(w);
        }
      }
    }
    else {
      // uplo == Uplo::Lower
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i=0 ; i<m ; ++i) {
          Td w;
          mixedp_mul<Td, Tc, Tscalar>(w, C(i, j), beta);
          Td s = Td_zero;
          for (idx_int k = 0; k <= j; ++k) {
            mixedp_madd<Td, Tb, Ta>(s, B(i, k), A(j, k));
          }
          for (idx_int k = j+1 ; k<n ; ++k) {
            mixedp_madd<Td, Tb, Ta>(s, B(i, k), A(k, j));
          }
          mixedp_madd<Td, Tscalar, Td>(w, alpha, s);
          C(i, j) = type_conv<Tc, Td>(w);
        }
      }
    }
  }

  #undef A
  #undef B
  #undef C
}

}
#endif
