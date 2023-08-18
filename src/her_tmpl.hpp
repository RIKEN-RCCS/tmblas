//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _HER_TMPL_HPP
#define _HER_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Hermitian matrix rank-1 update:
/// \[
///     A = \alpha x x^H + A,
/// \]
/// where alpha is a scalar, x is a vector,
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
/// @param[in, out] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @ingroup her

template<typename Ta, typename Tb, typename Td>
void her(
    blas::Uplo uplo,
    idx_int n,
    blas::real_type<Ta, Tb> const &alpha,
    Ta const *x, idx_int incx,
    Tb       *A, idx_int lda)
{
//  typedef blas::scalar_type<Ta, Tb> Tscalar;
  typedef blas::real_type<Ta, Tb> Tscalarreal;
  typedef blas::real_type<Td> Tdreal;
  typedef blas::real_type<Tb> Tbreal;  
  #define A(i_, j_) A[ (i_) + (j_)*lda ]

  // constants
  //  const Tscalar Tscalar_zero(Tscalarreal(0));
  //  const Tscalarreal Tscalarreal_zero(0);
  //  const Tdreal Tdreal_zero(0);
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper );
  blas_error_if( n < 0 );
  blas_error_if( incx == 0 );
  blas_error_if( lda < n );

  // quick return
  if (n == 0 || mixedp_eq<Tscalarreal, int>(alpha, 0)) {
    return;
  }

  idx_int kx = (incx > 0 ? 0 : (-n + 1)*incx);
  Td tmp, tmp2, tmp3;
  if (uplo == blas::Uplo::Upper) {
    if (incx == 1) {
      // unit stride
      for (idx_int j = 0; j < n; ++j) {
        // note: NOT skipping if x[j] is zero, for consistent NAN handling
        mixedp_mul<Td, Tscalarreal, Ta>(tmp, alpha, conjg<Ta>(x[j]));
        //scalar_t tmp = alpha * x[j];
        for (idx_int i = 0; i < j; ++i) {
          tmp2 = type_conv<Td, Tb>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp2, x[i], tmp);
          A(i, j) = type_conv<Tb, Td>(tmp2);
          //A(i, j) += x[i] * tmp;
        }
	{ // i == j
#if 0
	  mixedp_mul<Td, Ta, Ta>(tmp, x[j], conjg<Ta>(x[j]));
	  Tdreal tmp1, tmp4;
	  mixedp_mul<Tdreal, Tscalarreal, Tdreal>(tmp1, alpha, real<Tdreal>(tmp));
	  Tbreal tmp5(real<Tbreal>(A(j, j)));
	  //	  tmp2 = type_conv<Td, Tb>(A(j, j));
	  mixedp_add<Tdreal, Tbreal, Tdreal>(tmp4, tmp5, tmp1);
	  A(j, j) = type_conv<Tb, Td>(tmp4);
#else
	  mixedp_mul<Td, Ta, Ta>(tmp, x[j], conjg<Ta>(x[j]));
	  Tdreal tmp5;
	  tmp5 = type_conv<Tdreal, Tbreal>(real<Tbreal>(A(j, j)));
	  mixedp_madd<Tdreal, Tscalarreal, Tdreal>(tmp5, alpha, real<Tdreal>(tmp));
	  A(j, j) = type_conv<Tb, Tdreal>(tmp5);
#endif
	}
      }
    }
    else {
      // non-unit stride
      idx_int jx = kx;
      for (idx_int j = 0; j < n; ++j) {
        mixedp_mul<Td, Tscalarreal, Ta>(tmp, alpha, conjg<Ta>(x[jx]));
        //scalar_t tmp = alpha * x[jx];
        idx_int ix = kx;
        for (idx_int i = 0; i < j; ++i) {
          tmp2 = type_conv<Td, Tb>(A(i, j));
          mixedp_madd<Td, Ta, Td>(tmp2, x[ix], tmp);
          A(i, j) = type_conv<Tb, Td>(tmp2);
          //A(i, j) += x[ix] * tmp;
          ix += incx;
        }
	{ // i == j
#if 0
	  mixedp_mul<Td, Ta, Ta>(tmp, x[jx], conjg<Ta>(x[jx]));
	  Tdreal tmp1, tmp4;
	  mixedp_mul<Tdreal, Tscalarreal, Tdreal>(tmp1, alpha, real<Tdreal>(tmp));
	  Tbreal tmp5(real<Tbreal>(A(j, j)));
	  //	  tmp2 = type_conv<Td, Tb>(A(j, j));
	  mixedp_add<Tdreal, Tbreal, Tdreal>(tmp4, tmp5, tmp1);
	  A(j, j) = type_conv<Tb, Td>(tmp4);
#else
	  mixedp_mul<Td, Ta, Ta>(tmp, x[jx], conjg<Ta>(x[jx]));
	  Tdreal tmp5;
	  tmp5 = type_conv<Tdreal, Tbreal>(real<Tbreal>(A(j, j)));
	  mixedp_madd<Tdreal, Tscalarreal, Tdreal>(tmp5, alpha, real<Tdreal>(tmp));
	  A(j, j) = type_conv<Tb, Tdreal>(tmp5);	  
#endif
	  //	  
	  //	  tmp2 = type_conv<Td, Tb>(real<Tbreal>(A(j, j)));
	  //	  mixedp_add<Td, Td, Td>(tmp2, tmp2, tmp1);
	  //          A(j, j) = type_conv<Tb, Td>(tmp2);
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
        mixedp_mul<Td, Tscalarreal, Ta>(tmp, alpha, conjg<Ta>(x[j]));
        //scalar_t tmp = alpha * x[j];
	{ // i == j
#if 0
	  mixedp_mul<Td, Ta, Ta>(tmp3, x[j], conjg<Ta>(x[j]));
	  Tdreal tmp1, tmp4;
	  mixedp_mul<Tdreal, Tscalarreal, Tdreal>(tmp1, alpha, real<Tdreal>(tmp3));
	  Tbreal tmp5(real<Tbreal>(A(j, j)));
	  mixedp_add<Tdreal, Tbreal, Tdreal>(tmp4, tmp5, tmp1);
          A(j, j) = type_conv<Tb, Td>(tmp4);
#else
	  mixedp_mul<Td, Ta, Ta>(tmp3, x[j], conjg<Ta>(x[j]));
	  Tdreal tmp5;
	  tmp5 = type_conv<Tdreal, Tbreal>(real<Tbreal>(A(j, j)));
	  mixedp_madd<Tdreal, Tscalarreal, Tdreal>(tmp5, alpha, real<Tdreal>(tmp3));
	  A(j, j) = type_conv<Tb, Tdreal>(tmp5);	  
#endif
	}
        for (idx_int i = j + 1; i < n; ++i) {
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
        mixedp_mul<Td, Tscalarreal, Ta>(tmp, alpha, conjg<Ta>(x[jx]));

        //scalar_t tmp = alpha * x[jx];
        idx_int ix = jx;
	{  // i == j
#if 0
	  mixedp_mul<Td, Ta, Ta>(tmp3, x[jx], conjg<Ta>(x[jx]));
	  Tdreal tmp1, tmp4;
	  mixedp_mul<Tdreal, Tscalarreal, Tdreal>(tmp1, alpha, real<Tdreal>(tmp3));
	  Tbreal tmp5(real<Tbreal>(A(j, j)));
	      //	  tmp2 = type_conv<Td, Tb>(A(j, j));
	  mixedp_add<Tdreal, Tbreal, Tdreal>(tmp4, tmp5, tmp1);
          A(j, j) = type_conv<Tb, Td>(tmp4);
#else
	  mixedp_mul<Td, Ta, Ta>(tmp3, x[jx], conjg<Ta>(x[jx]));
	  Tdreal tmp5;
	  tmp5 = type_conv<Tdreal, Tbreal>(real<Tbreal>(A(j, j)));
	  mixedp_madd<Tdreal, Tscalarreal, Tdreal>(tmp5, alpha, real<Tdreal>(tmp3));
	  A(j, j) = type_conv<Tb, Tdreal>(tmp5);	  
#endif
	  ix += incx;
	}
        for (idx_int i = j + 1; i < n; ++i) {
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

