//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TRSM_TMPL_HPP
#define _TRSM_TMPL_HPP

namespace tmblas{

// =============================================================================
/// Solve the triangular matrix-vector equation
/// \[
///     op(A) X = \alpha B,
/// \]
/// or
/// \[
///     X op(A) = \alpha B,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// X and B are m-by-n matrices, and A is an m-by-m or n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// @see latrs for a more numerically robust implementation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] side
///     Whether $op(A)$ is on the left or right of X:
///     - Side::Left:  $op(A) X = B$.
///     - Side::Right: $X op(A) = B$.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero:
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///
/// @param[in] trans
///     The form of $op(A)$:
///     - Op::NoTrans:   $op(A) = A$.
///     - Op::Trans:     $op(A) = A^T$.
///     - Op::ConjTrans: $op(A) = A^H$.
///
/// @param[in] diag
///     Whether A has a unit or non-unit diagonal:
///     - Diag::Unit:    A is assumed to be unit triangular.
///     - Diag::NonUnit: A is not assumed to be unit triangular.
///
/// @param[in] m
///     Number of rows of matrices B and X. m >= 0.
///
/// @param[in] n
///     Number of columns of matrices B and X. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not accessed.
///
/// @param[in] A
///     - If side = Left:
///       the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
///     - If side = Right:
///       the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If side = left:  lda >= max(1, m).
///     - If side = right: lda >= max(1, n).
///
/// @param[in, out] B
///     On entry,
///     the m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
///     On exit, overwritten by the solution matrix X.
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
///
/// @param w
///     Work array of size column vector : m of B or row vector : n of B^T
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup trsm

template< typename Ta, typename Tb, typename Td >
void trsm(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    blas::scalar_type<Ta, Tb> const &alpha,
    Ta const *A, idx_int lda,
    Tb       *B, idx_int ldb,
    Td       *w)
{
  typedef blas::scalar_type<Ta, Tb> Tscalar;
  typedef blas::real_type<Tscalar> Tscalarreal;
  typedef blas::real_type<Tb> Tbreal;  
  bool hasworking = (!(w == (Td *)nullptr));
  Td *ww;
 
  const Tb Tb_zero(Tbreal(0));
  const Tscalar Tscalar_zero(Tscalarreal(0));
  #define A(i_, j_) A[ (i_) + (j_)*lda ]
  #define B(i_, j_) B[ (i_) + (j_)*ldb ]

  // check arguments
  blas_error_if( side != blas::Side::Left &&
                 side != blas::Side::Right );
  blas_error_if( uplo != blas::Uplo::Lower &&
                 uplo != blas::Uplo::Upper );
  blas_error_if( trans != blas::Op::NoTrans &&
                 trans != blas::Op::Trans &&
                 trans != blas::Op::ConjTrans );
  blas_error_if( diag != blas::Diag::NonUnit &&
                 diag != blas::Diag::Unit );
  blas_error_if( m < 0 );
  blas_error_if( n < 0 );

  // check remaining arguments
  blas_error_if( lda < ((side == blas::Side::Left) ? m : n) );
  blas_error_if( ldb < m );

  // quick return
  if (m == 0 || n == 0) {
    return;
  }

  if (!hasworking) {
    int matsize = (side  == blas::Side::Left) ? m : n;
    ww = new Td[matsize];
  }
  else {
    ww = w;
  }
  // alpha == zero
  if (alpha == Tscalar_zero) {
    for (idx_int j = 0; j < n; ++j) {
      for (idx_int i = 0; i < m; ++i) {
        B(i, j) = Tb_zero;
      }
    }
    return;
  }
  if (side == blas::Side::Left) {

    if (trans == blas::Op::NoTrans) {
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < m; ++i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = 0; i < m; ++i) {
            mixedp_mul<Td, Td, Tscalar>(ww[i], ww[i], alpha);
            //B(i, j) *= alpha;
          }
          for (idx_int k = m-1; k >= 0; --k) {
            if (diag == blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[k], ww[k], A(k, k));
              //B(k, j) /= A(k, k);
            }
            for (idx_int i = 0; i < k; ++i) {
              mixedp_msub<Td, Ta, Td>(ww[i], A(i, k), ww[k]);
              //B(i, j) -= A(i, k)*B(k, j);
            }
          }
          for (idx_int i = 0; i < m; ++i) {
            B(i, j) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < m; ++i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = 0; i < m; ++i) {
            mixedp_mul<Td, Td, Tscalar>(ww[i], ww[i], alpha);
            //B(i, j) *= alpha;
          }
          for (idx_int k = 0; k < m; ++k) {
            if (diag == blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[k], ww[k], A(k, k));
              //B(k, j) /= A(k, k);
            }
            for (idx_int i = k+1; i < m; ++i) {
              mixedp_msub<Td, Ta, Td>(ww[i], A(i, k), ww[k]);
              //B(i, j) -= A(i, k)*B(k, j);
            }
          }
          for (idx_int i = 0; i < m; ++i) {
            B(i, j) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
    }
    else if (trans == blas::Op::Trans) {
      Td sum;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < m; ++i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = 0; i < m; ++i) {
            mixedp_mul<Td, Tscalar, Td>(sum, alpha, ww[i]);
            //scalar_t sum = alpha*B(i, j);
            for (idx_int k = 0; k < i; ++k) {
              mixedp_msub<Td, Ta, Td>(sum, A(k, i), ww[k]);
              //sum -= A(k, i)*B(k, j);
            }
            if(diag==blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[i], sum, A(i, i));
            } else {
              ww[i] = sum;
            }
            //B(i, j) = (diag == Diag::NonUnit)
            //          ? sum / A(i, i)
            //          : sum;
          }
          for (idx_int i = 0; i < m; ++i) {
            B(i, j) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < m; ++i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = m-1; i >= 0; --i) {
            mixedp_mul<Td, Tscalar, Td>(sum, alpha, ww[i]);
            //scalar_t sum = alpha*B(i, j);
            for (idx_int k = i+1; k < m; ++k) {
              mixedp_msub<Td, Ta, Td>(sum, A(k, i), ww[k]);
              //sum -= A(k, i)*B(k, j);
            }
            if(diag==blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[i], sum, A(i, i));
            } else {
              ww[i] = sum;
            }
            //B(i, j) = (diag == Diag::NonUnit)
            //          ? sum / A(i, i)
            //          : sum;
          }
          for (idx_int i = 0; i < m; ++i) {
            B(i, j) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
    }
    else { // trans == Op::ConjTrans
      Td sum;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < m; ++i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = 0; i < m; ++i) {
            mixedp_mul<Td, Tscalar, Td>(sum, alpha, ww[i]);
            //scalar_t sum = alpha*B(i, j);
            for (idx_int k = 0; k < i; ++k) {
              mixedp_msub<Td, Ta, Td>(sum, conjg<Ta>(A(k, i)), ww[k]);
              //sum -= conj(A(k, i))*B(k, j);
            }
            if(diag==blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[i], sum, conjg<Ta>(A(i, i)));
            } else {
              ww[i] = sum;
            }
            //B(i, j) = (diag == Diag::NonUnit)
            //          ? sum / conj(A(i, i))
            //          : sum;
          }
          for (idx_int i = 0; i < m; ++i) {
            B(i, j) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = 0; i < m; ++i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = m-1; i >= 0; --i) {
            mixedp_mul<Td, Tscalar, Td>(sum, alpha, ww[i]);
            //scalar_t sum = alpha*B(i, j);
            for (idx_int k = i+1; k < m; ++k) {
              mixedp_msub<Td, Ta, Td>(sum, conjg<Ta>(A(k, i)), ww[k]);
              //sum -= conj(A(k, i))*B(k, j);
            }
            if(diag==blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[i], sum, conjg<Ta>(A(i, i)));
            } else {
              ww[i] = sum;
            }
            //B(i, j) = (diag == Diag::NonUnit)
            //          ? sum / conj(A(i, i))
            //          : sum;
          }
          for (idx_int i = 0; i < m; ++i) {
            B(i, j) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
    }
  }
  else { // side == Side::Right
    if (trans == blas::Op::Trans) {
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = 0; i < n; ++i) {
            mixedp_mul<Td, Td, Tscalar>(ww[i], ww[i], alpha);
            //B(j, i) *= alpha;
          }
          for (idx_int k = n-1; k >= 0; --k) {
            if (diag == blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[k], ww[k], A(k, k));
              //B(j, k) /= A(k, k);
            }
            for (idx_int i = 0; i < k; ++i) {
              mixedp_msub<Td, Ta, Td>(ww[i], A(i, k), ww[k]);
              //B(j, i) -= A(i, k)*B(k, j);
            }
          }
          for (idx_int i = 0; i < n; ++i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = 0; i < n; ++i) {
            mixedp_mul<Td, Td, Tscalar>(ww[i], ww[i], alpha);
            //B(j, i) *= alpha;
          }
          for (idx_int k = 0; k < n; ++k) {
            if (diag == blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[k], ww[k], A(k, k));
              //B(j, i) /= A(k, k);
            }
            for (idx_int i = k+1; i < n; ++i) {
              mixedp_msub<Td, Ta, Td>(ww[i], A(i, k), ww[k]);
              //B(j, i) -= A(i, k)*B(j, k);
            }
          }
          for (idx_int i = 0; i < n; ++i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
    }
    else if (trans == blas::Op::NoTrans) {
      Td sum;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = 0; i < n; ++i) {
            mixedp_mul<Td, Tscalar, Td>(sum, alpha, ww[i]);
            //scalar_t sum = alpha*B(j, i);
            for (idx_int k = 0; k < i; ++k) {
              mixedp_msub<Td, Ta, Td>(sum, A(k, i), ww[k]);
              //sum -= A(k, i)*B(j, k);
            }
            if(diag==blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[i], sum, A(i, i));
            } else {
              ww[i] = sum;
            }
            //B(j, i) = (diag == Diag::NonUnit)
            //          ? sum / A(i, i)
            //          : sum;
          }
          for (idx_int i = 0; i < n; ++i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = n-1; i >= 0; --i) {
            mixedp_mul<Td, Tscalar, Td>(sum, alpha, ww[i]);
            //scalar_t sum = alpha*B(j, i);
            for (idx_int k = i+1; k < n; ++k) {
              mixedp_msub<Td, Ta, Td>(sum, A(k, i), ww[k]);
              //sum -= A(k, i)*B(j, k);
            }
            if(diag==blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[i], sum, A(i, i));
            } else {
              ww[i] = sum;
            }
            //B(j, i) = (diag == Diag::NonUnit)
            //          ? sum / A(i, i)
            //          : sum;
          }
          for (idx_int i = 0; i < n; ++i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
    }
    else { // trans == Op::ConjTrans
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = 0; i < n; ++i) {
            mixedp_mul<Td, Td, Tscalar>(ww[i], ww[i], alpha);
            //B(j, i) *= alpha;
          }
          for (idx_int k = n-1; k >= 0; --k) {
            if (diag == blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[k], ww[k], conjg<Ta>(A(k, k)));
              //B(j, k) /= A(k, k);
            }
            for (idx_int i = 0; i < k; ++i) {
              mixedp_msub<Td, Ta, Td>(ww[i], conjg<Ta>(A(i, k)), ww[k]);
              //B(j, i) -= A(k, i)*B(k, j);
            }
          }
          for (idx_int i = 0; i < n; ++i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = 0; i < n; ++i) {
            mixedp_mul<Td, Td, Tscalar>(ww[i], ww[i], alpha);
            //B(j, i) *= alpha;
          }
          for (idx_int k = 0; k < n; ++k) {
            if (diag == blas::Diag::NonUnit) {
              mixedp_div<Td, Td, Ta>(ww[k], ww[k], conjg<Ta>(A(k, k)));
              //B(j, i) /= A(k, k);
            }
            for (idx_int i = k+1; i < n; ++i) {
              mixedp_msub<Td, Ta, Td>(ww[i], conjg<Ta>(A(i, k)), ww[k]);
              //B(j, i) -= A(k, i)*B(j, k);
            }
          }
          for (idx_int i = 0; i < n; ++i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
	}
      }
    }
  }

  if(!hasworking) {
    delete[] ww;
  }

  #undef A
  #undef B
}

}
#endif
