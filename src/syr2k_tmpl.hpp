//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SYR2K_TMPL_HPP
#define _SYR2K_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Symmetric rank-k update:
/// \[
///     C = \alpha A B^T + \alpha B A^T + \beta C,
/// \]
/// or
/// \[
///     C = \alpha A^T B + \alpha B^T A + \beta C,
/// \]
/// where alpha and beta are scalars, C is an n-by-n symmetric matrix,
/// and A and B are n-by-k or k-by-n matrices.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] uplo
///     What part of the matrix C is referenced,
///     the opposite triangle being assumed from symmetry:
///     - Uplo::Lower: only the lower triangular part of C is referenced.
///     - Uplo::Upper: only the upper triangular part of C is referenced.
///
/// @param[in] trans
///     The operation to be performed:
///     - Op::NoTrans: $C = \alpha A B^T + \alpha B A^T + \beta C$.
///     - Op::Trans:   $C = \alpha A^T B + \alpha B^T A + \beta C$.
///     - In the real    case, Op::ConjTrans is interpreted as Op::Trans.
///       In the complex case, Op::ConjTrans is illegal (see @ref her2k instead).
///
/// @param[in] n
///     Number of rows and columns of the matrix C. n >= 0.
///
/// @param[in] k
///     - If trans = NoTrans: number of columns of the matrix A. k >= 0.
///     - Otherwise:          number of rows    of the matrix A. k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If trans = NoTrans:
///       the n-by-k matrix A, stored in an lda-by-k array [RowMajor: n-by-lda].
///     - Otherwise:
///       the k-by-n matrix A, stored in an lda-by-n array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If trans = NoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)],
///     - Otherwise:          lda >= max(1, k) [RowMajor: lda >= max(1, n)].
///
/// @param[in] B
///     - If trans = NoTrans:
///       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
///     - Otherwise:
///       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If trans = NoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)],
///     - Otherwise:          ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The n-by-n symmetric matrix C,
///     stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, n).
///
/// @param w
///     Work array of size n.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup syr2k

template< typename Ta, typename Tb, typename Tc, typename Td >
void syr2k(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    blas::scalar_type<Ta, Tb, Tc> const &alpha,
    Ta const *A, idx_int lda,
    Tb const *B, idx_int ldb,
    blas::scalar_type<Ta, Tb, Tc> const &beta,
    Tc       *C, idx_int ldc,
    Td       *w)
{
  typedef blas::scalar_type<Ta, Tb, Tc> Tscalar;
  typedef blas::real_type<Tscalar> Tscalarreal;
  typedef blas::real_type<Tc> Tcreal;
  typedef blas::real_type<Td> Tdreal;    

  bool hasworking = (!(w == (Td *)nullptr) &&
		     (trans == blas::Op::NoTrans));
  const Tscalar Tscalar_zero(Tscalarreal(0)), Tscalar_one(Tscalarreal(1));
  const Tc Tc_zero(Tcreal(0));
  Td *ww;
  const Td Td_zero(Tdreal(0));

  #define A(i_, j_) A[ (i_) + (j_)*lda ]
  #define B(i_, j_) B[ (i_) + (j_)*ldb ]
  #define C(i_, j_) C[ (i_) + (j_)*ldc ]

  // check arguments
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper &&
                 uplo != blas::Uplo::General );
  blas_error_if( n < 0 );
  blas_error_if( k < 0 );

  // check and interpret argument trans
  if (trans == blas::Op::ConjTrans) {
    blas_error_if_msg(
            ( blas::is_complex<Ta>::value ||
              blas::is_complex<Tb>::value ),
            "trans == Op::ConjTrans &&"
            "(is_complex<TA>::value ||"
            " is_complex<TB>::value)" );
    trans = blas::Op::Trans;
  }
  else {
    blas_error_if( trans != blas::Op::NoTrans &&
                   trans != blas::Op::Trans );
  }

  // check remaining arguments
  blas_error_if( lda < ((trans == blas::Op::NoTrans) ? n : k) );
  blas_error_if( ldb < ((trans == blas::Op::NoTrans) ? n : k) );
  blas_error_if( ldc < n );

  // quick return
  if (n == 0 || k == 0) {
    return;
  }

  // alpha == zero
  if (alpha == Tscalar_zero) {
    if (beta == Tscalar_zero) {
      if (uplo != blas::Uplo::Upper) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i <= j; ++i) {
            C(i, j) = Tc_zero;
          }
        }
      }
      else if (uplo != blas::Uplo::Lower) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = j; i < n; ++i) {
            C(i, j) = Tc_zero;
          }
        }
      }
      else {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            C(i, j) = Tc_zero;
          }
        }
      }
    }
    else if (beta != Tscalar_one) {
      Td tmp;
      if (uplo != blas::Uplo::Upper) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i <= j; ++i) {
            mixedp_mul<Td, Tc, Tscalar>(tmp, C(i, j), beta);
            C(i, j) = type_conv<Tc, Td>(tmp);
          }
        }
      }
      else if (uplo != blas::Uplo::Lower) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = j; i < n; ++i) {
            mixedp_mul<Td, Tc, Tscalar>(tmp, C(i, j), beta);
            C(i, j) = type_conv<Tc, Td>(tmp);
          }
        }
      }
      else {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            mixedp_mul<Td, Tc, Tscalar>(tmp, C(i, j), beta);
            C(i, j) = type_conv<Tc, Td>(tmp);
          }
        }
      }
    }
    return;
  }

  if (!hasworking) {
    ww = new Td[n];
  } 
  else {
    ww = w;
  }


    // alpha != zero
  if (trans == blas::Op::NoTrans) {
    Td alpha_Bjl, alpha_Ajl;
    if (uplo != blas::Uplo::Lower) {
      // uplo == Uplo::Upper or uplo == Uplo::General
      for (idx_int j = 0; j < n; ++j) {

        for (idx_int i = 0; i <= j; ++i) {
          mixedp_mul<Td, Tc, Tscalar>(ww[i], C(i, j), beta);
          //C(i, j) *= beta;
        }

        for (idx_int l = 0; l < k; ++l) {
          mixedp_mul<Td, Tscalar, Tb>(alpha_Bjl, alpha, B(j, l));
          mixedp_mul<Td, Tscalar, Ta>(alpha_Ajl, alpha, A(j, l));
          //scalar_t alpha_Bjl = alpha*B(j, l);
          //scalar_t alpha_Ajl = alpha*A(j, l);
          for (idx_int i = 0; i <= j; ++i) {
            mixedp_madd<Td, Ta, Td>(ww[i], A(i, l), alpha_Bjl);
            mixedp_madd<Td, Tb, Td>(ww[i], B(i, l), alpha_Ajl);
            //ww[i] += tmp1+tmp2;
            //C(i, j) += A(i, l)*alpha_Bjl + B(i, l)*alpha_Ajl;
          }
        }
        for (idx_int i = 0; i <= j; ++i) {
          C(i, j) = type_conv<Tc, Td>(ww[i]);
        }
      }
    }
    else { // uplo == Uplo::Lower
      for (idx_int j = 0; j < n; ++j) {

        for (idx_int i = j; i < n; ++i) {
          mixedp_mul<Td, Tc, Tscalar>(ww[i], C(i, j), beta);
          //C(i, j) *= beta;
        }

        for (idx_int l = 0; l < k; ++l) {
          mixedp_mul<Td, Tscalar, Tb>(alpha_Bjl, alpha, B(j, l));
          mixedp_mul<Td, Tscalar, Ta>(alpha_Ajl, alpha, A(j, l));
          //scalar_t alpha_Bjl = alpha*B(j, l);
          //scalar_t alpha_Ajl = alpha*A(j, l);
          for (idx_int i = j; i < n; ++i) {
            mixedp_madd<Td, Ta, Td>(ww[i], A(i, l), alpha_Bjl);
            mixedp_madd<Td, Tb, Td>(ww[i], B(i, l), alpha_Ajl);
            //ww[i] += tmp1+tmp2;
            //C(i, j) += A(i, l)*alpha_Bjl + B(i, l)*alpha_Ajl;
          }
        }
        for (idx_int i = j; i < n; ++i) {
          C(i, j) = type_conv<Tc, Td>(ww[i]);
        }
      }
    }
  }
  else { // trans == Op::Trans
    Td sum1, sum2, tmp1;
    if (uplo != blas::Uplo::Lower) {
      // uplo == Uplo::Upper or uplo == Uplo::General
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i = 0; i <= j; ++i) {
          sum1 = Td_zero;
          sum2 = Td_zero;
          //scalar_t sum1 = zero;
          //scalar_t sum2 = zero;
          for (idx_int l = 0; l < k; ++l) {
            mixedp_madd<Td, Ta, Tb>(sum1, A(l, i), B(l, j));
            mixedp_madd<Td, Tb, Ta>(sum2, B(l, i), A(l, j));
            //sum1 += A(l, i) * B(l, j);
            //sum2 += B(l, i) * A(l, j);
          }
          mixedp_mul<Td, Tscalar, Tc>(tmp1, beta, C(i, j)); 
          mixedp_madd<Td, Tscalar, Td>(tmp1, alpha, sum1);
          mixedp_madd<Td, Tscalar, Td>(tmp1, alpha, sum2);
          C(i, j) = type_conv<Tc, Td>(tmp1);
          //C(i, j) = type_conv<Tc, Td>(tmp1+tmp2+tmp3);
          //C(i, j) = alpha*sum1 + alpha*sum2 + beta*C(i, j);
        }
      }
    }
    else { // uplo == Uplo::Lower
      for (idx_int j = 0; j < n; ++j) {
        for (idx_int i = j; i < n; ++i) {
          sum1 = Td_zero;
          sum2 = Td_zero;
          //scalar_t sum1 = zero;
          //scalar_t sum2 = zero;
          for (idx_int l = 0; l < k; ++l) {
            mixedp_madd<Td, Ta, Tb>(sum1, A(l, i), B(l, j));
            mixedp_madd<Td, Tb, Ta>(sum2, B(l, i), A(l, j));
            //sum1 +=  A(l, i) * B(l, j);
            //sum2 +=  B(l, i) * A(l, j);
          }
          mixedp_mul<Td, Tscalar, Tc>(tmp1, beta, C(i, j));
          mixedp_madd<Td, Tscalar, Td>(tmp1, alpha, sum1);
          mixedp_madd<Td, Tscalar, Td>(tmp1, alpha, sum2);
          C(i, j) = type_conv<Tc, Td>(tmp1);
          //C(i, j) = type_conv<Tc, Td>(tmp1+tmp2+tmp3);
          //C(i, j) = alpha*sum1 + alpha*sum2 + beta*C(i, j);
        }
      }
    }
  }

  if (uplo == blas::Uplo::General) {
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t i = j+1; i < n; ++i) {
        C(i, j) = C(j, i);
      }
    }
  }

  if(!hasworking) {
    delete [] ww;
  }

  #undef A
  #undef B
  #undef C
}

}
#endif
