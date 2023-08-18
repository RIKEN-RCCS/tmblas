//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TRMM_TMPL_HPP
#define _TRMM_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Triangular matrix-matrix multiply:
/// \[
///     B = \alpha op(A) B,
/// \]
/// or
/// \[
///     B = \alpha B op(A),
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// B is an m-by-n matrix, and A is an m-by-m or n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] side
///     Whether $op(A)$ is on the left or right of B:
///     - Side::Left:  $B = \alpha op(A) B$.
///     - Side::Right: $B = \alpha B op(A)$.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero:
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///     - Uplo::General is illegal (see @ref gemm instead).
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
///     Number of rows of matrix B. m >= 0.
///
/// @param[in] n
///     Number of columns of matrix B. n >= 0.
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
///     The m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
///
/// @param w
///     Work array of size column vector : m of B or row vector : n of B^T
///     Will be allocated/deallocated within the function if not given.

/// @ingroup trmm

template< typename Ta, typename Tb, typename Td>
void trmm(
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
  const Tb Tb_zero(Tbreal(0));
  const Tscalar Tscalar_zero(Tscalarreal(0));
  Td *ww;
  #define A(i_, j_) A[ (i_) + (j_)*lda ]
  #define B(i_, j_) B[ (i_) + (j_)*ldb ]
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

  // alpha == zero
  if (alpha == Tscalar_zero) {
    for (idx_int j = 0; j < n; ++j) {
      for (idx_int i = 0; i < m; ++i) {
        B(i, j) =Tb_zero;
      }
    }
    return;
  }

  if (!hasworking) {
    int matsize = (side  == blas::Side::Left) ? m : n;
    ww = new Td[matsize];
  }
  else {
    ww = w;
  }

  if (side == blas::Side::Left) {
    if (trans == blas::Op::NoTrans) {
      Td alpha_Bkj;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < n; ++j) {
          for(idx_int k=0 ; k<m ; ++k) {
            ww[k] = type_conv<Td, Tb>(B(k, j));
          }
          for (idx_int k = 0; k < m; ++k) {
            mixedp_mul<Td, Tscalar, Td>(alpha_Bkj, alpha, ww[k]);
            //scalar_t alpha_Bkj = alpha*B(k, j);
            for (idx_int i = 0; i < k; ++i) {
              mixedp_madd<Td, Ta, Td>(ww[i], A(i, k), alpha_Bkj);
              //B(i, j) += A(i, k)*alpha_Bkj;
            }
            if (diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(ww[k], A(k, k), alpha_Bkj);
            }
            else {
              ww[k] = alpha_Bkj;
            }
            //B(k, j) = (diag == Diag::NonUnit)
            //          ? A(k, k)*alpha_Bkj
            //          : alpha_Bkj;
          }
          for(idx_int k=0 ; k<m ; ++k) {
            B(k, j) = type_conv<Tb, Td>(ww[k]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int k = m-1; k >= 0; --k) {
            ww[k] = type_conv<Td, Tb>(B(k, j));
          }
          for (idx_int k = m-1; k >= 0; --k) {
            mixedp_mul<Td, Tscalar, Td>(alpha_Bkj, alpha, ww[k]);
            if (diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(ww[k], A(k, k), alpha_Bkj);
            }
            else {
              ww[k] = alpha_Bkj;
            }
            //scalar_t alpha_Bkj = alpha*B(k, j);
            //B(k, j) = (diag == Diag::NonUnit)
            //          ? A(k, k)*alpha_Bkj
            //          : alpha_Bkj;
            for (idx_int i = k+1; i < m; ++i) {
              mixedp_madd<Td, Ta, Td>(ww[i], A(i, k), alpha_Bkj);
              //B(i, j) += A(i, k)*alpha_Bkj;
            }
          }
          for (idx_int k = m-1; k >= 0; --k) {
            B(k, j) = type_conv<Tb, Td>(ww[k]);
          }
        }
      }
    }
    else if (trans == blas::Op::Trans) {
      Td sum;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < n; ++j) {
          for (idx_int i = m-1; i >= 0; --i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = m-1; i >= 0; --i) {
            if(diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(sum, A(i, i), ww[i]);
            }
            else{
              sum = ww[i];
//            scalar_t sum = (diag == Diag::NonUnit)
//                           ? A(i, i)*B(i, j)
//                           : B(i, j);
            }
            for (idx_int k = 0; k < i; ++k) {
              mixedp_madd<Td, Ta, Td>(sum, A(k, i), ww[k]);
                //sum += A(k, i)*B(k, j);
            }
            mixedp_mul<Td, Tscalar, Td>(ww[i], alpha, sum);
            //B(i, j) = alpha * sum;
          }
          for (idx_int i = m-1; i >= 0; --i) {
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
            if(diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(sum, A(i, i), ww[i]);
            }
            else{
              sum = ww[i];
            //scalar_t sum = (diag == Diag::NonUnit)
            //               ? A(i, i)*B(i, j)
            //               : B(i, j);
            }
            for (idx_int k = i+1; k < m; ++k) {
              mixedp_madd<Td, Ta, Td>(sum, A(k, i), ww[k]);
              //sum += A(k, i)*B(k, j);
            }
            mixedp_mul<Td, Tscalar, Td>(ww[i], alpha, sum);
            //B(i, j) = alpha * sum;
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
          for (idx_int i = m-1; i >= 0; --i) {
            ww[i] = type_conv<Td, Tb>(B(i, j));
          }
          for (idx_int i = m-1; i >= 0; --i) {
            if(diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(sum, conjg<Ta>(A(i, i)), ww[i]);
            }
            else{
              sum = ww[i];
            //scalar_t sum = (diag == Diag::NonUnit)
            //               ? conj(A(i, i))*B(i, j)
            //               : B(i, j);
            }
            for (idx_int k = 0; k < i; ++k) {
              mixedp_madd<Td, Ta, Td>(sum, conjg<Ta>(A(k, i)), ww[k]);
                //sum += conj(A(k, i))*B(k, j);
            }
            mixedp_mul<Td, Tscalar, Td>(ww[i], alpha, sum);
            //B(i, j) = alpha * sum;
          }
          for (idx_int i = m-1; i >= 0; --i) {
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
            if(diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(sum, conjg<Ta>(A(i, i)), ww[i]);
            }
            else {
              sum = ww[i];
            //scalar_t sum = (diag == Diag::NonUnit)
            //               ? conj(A(i, i))*B(i, j)
            //               : B(i, j);
            }
            for (idx_int k = i+1; k < m; ++k) {
              mixedp_madd<Td, Ta, Td>(sum, conjg<Ta>(A(k, i)), ww[k]);
                //sum += conj(A(k, i))*B(k, j);
            }
            mixedp_mul<Td, Tscalar, Td>(ww[i], alpha, sum);
            //B(i, j) = alpha * sum;
          }
          for (idx_int i = 0; i < m; ++i) {
            B(i, j) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
    }
  }
  else { // side == Side::Right  :: differs from blaspp implementation
         //                         based on B^T = \alpha op(A)^T B^T
    if (trans == blas::Op::Trans) {
      Td alpha_Bkj;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < m; ++j) {
          for(idx_int k=0 ; k<n ; ++k) {
            ww[k] = type_conv<Td, Tb>(B(j, k));
          }
          for (idx_int k = 0; k < n; ++k) {
            mixedp_mul<Td, Tscalar, Td>(alpha_Bkj, alpha, ww[k]);
            //scalar_t alpha_Bkj = alpha*B(k, j);
            for (idx_int i = 0; i < k; ++i) {
              mixedp_madd<Td, Ta, Td>(ww[i], A(i, k), alpha_Bkj);
              //B(i, j) += A(i, k)*alpha_Bkj;
            }
            if (diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(ww[k], A(k, k), alpha_Bkj);
            }
            else {
              ww[k] = alpha_Bkj;
            }
            //B(k, j) = (diag == Diag::NonUnit)
            //          ? A(k, k)*alpha_Bkj
            //          : alpha_Bkj;
          }
          for(idx_int k=0 ; k<n ; ++k) {
            B(j, k) = type_conv<Tb, Td>(ww[k]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int k = n-1; k >= 0; --k) {
            ww[k] = type_conv<Td, Tb>(B(j, k));
          }
          for (idx_int k = n-1; k >= 0; --k) {
            mixedp_mul<Td, Tscalar, Td>(alpha_Bkj, alpha, ww[k]);
            if (diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(ww[k], A(k, k), alpha_Bkj);
            }
            else {
              ww[k] = alpha_Bkj;
            }
            //scalar_t alpha_Bkj = alpha*B(k, j);
            //B(k, j) = (diag == Diag::NonUnit)
            //          ? A(k, k)*alpha_Bkj
            //          : alpha_Bkj;
            for (idx_int i = k+1; i < n; ++i) {  // A n-by-n matrix
              mixedp_madd<Td, Ta, Td>(ww[i], A(i, k), alpha_Bkj);
              //B(i, j) += A(i, k)*alpha_Bkj;
            }
          }
          for (idx_int k = n-1; k >= 0; --k) {
            B(j, k) = type_conv<Tb, Td>(ww[k]);
          }
        }
      }
    }
    else if (trans == blas::Op::NoTrans) {
      Td sum;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = n-1; i >= 0; --i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = n-1; i >= 0; --i) { // A n-by-n matrix
            if(diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(sum, A(i, i), ww[i]);
            }
            else{
              sum = ww[i];
//            scalar_t sum = (diag == Diag::NonUnit)
//                           ? A(i, i)*B(i, j)
//                           : B(i, j);
            }
            for (idx_int k = 0; k < i; ++k) {
              mixedp_madd<Td, Ta, Td>(sum, A(k, i), ww[k]);
                //sum += A(k, i)*B(k, j);
            }
            mixedp_mul<Td, Tscalar, Td>(ww[i], alpha, sum);
            //B(i, j) = alpha * sum;
          }
          for (idx_int i = n-1; i >= 0; --i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int i = 0; i < n; ++i) {
            ww[i] = type_conv<Td, Tb>(B(j, i));
          }
          for (idx_int i = 0; i < n; ++i) {  // A n-by-n matrix
            if(diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(sum, A(i, i), ww[i]);
            }
            else{
              sum = ww[i];
            //scalar_t sum = (diag == Diag::NonUnit)
            //               ? A(i, i)*B(i, j)
            //               : B(i, j);
            }
            for (idx_int k = i+1; k < n; ++k) { // A n-by-n matrix
              mixedp_madd<Td, Ta, Td>(sum, A(k, i), ww[k]);
              //sum += A(k, i)*B(k, j);
            }
            mixedp_mul<Td, Tscalar, Td>(ww[i], alpha, sum);
            //B(i, j) = alpha * sum;
          }
          for (idx_int i = 0; i < n; ++i) {
            B(j, i) = type_conv<Tb, Td>(ww[i]);
          }
        }
      }
    }
    else { // trans == Op::ConjTrans
      Td alpha_Bkj;
      if (uplo == blas::Uplo::Upper) {
        for (idx_int j = 0; j < m; ++j) {
          for(idx_int k=0 ; k<n ; ++k) {
            ww[k] = type_conv<Td, Tb>(B(j, k));
          }
          for (idx_int k = 0; k < n; ++k) {
            mixedp_mul<Td, Tscalar, Td>(alpha_Bkj, alpha, ww[k]);
            //scalar_t alpha_Bkj = alpha*B(k, j);
            for (idx_int i = 0; i < k; ++i) {
              mixedp_madd<Td, Ta, Td>(ww[i], conjg<Ta>(A(i, k)), alpha_Bkj);
              //B(i, j) += A(i, k)*alpha_Bkj;
            }
            if (diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(ww[k], conjg<Ta>(A(k, k)), alpha_Bkj);
            }
            else {
              ww[k] = alpha_Bkj;
            }
            //B(k, j) = (diag == Diag::NonUnit)
            //          ? A(k, k)*alpha_Bkj
            //          : alpha_Bkj;
          }
          for(idx_int k=0 ; k<n ; ++k) {
            B(j, k) = type_conv<Tb, Td>(ww[k]);
          }
        }
      }
      else { // uplo == Uplo::Lower
        for (idx_int j = 0; j < m; ++j) {
          for (idx_int k = n-1; k >= 0; --k) {
            ww[k] = type_conv<Td, Tb>(B(j, k));
          }
          for (idx_int k = n-1; k >= 0; --k) {
            mixedp_mul<Td, Tscalar, Td>(alpha_Bkj, alpha, ww[k]);
            if (diag == blas::Diag::NonUnit) {
              mixedp_mul<Td, Ta, Td>(ww[k], conjg<Ta>(A(k, k)), alpha_Bkj);
            }
            else {
              ww[k] = alpha_Bkj;
            }
            //scalar_t alpha_Bkj = alpha*B(k, j);
            //B(k, j) = (diag == Diag::NonUnit)
            //          ? A(k, k)*alpha_Bkj
            //          : alpha_Bkj;
            for (idx_int i = k+1; i < n; ++i) {  // A n-by-n matrix
              mixedp_madd<Td, Ta, Td>(ww[i], conjg<Ta>(A(i, k)), alpha_Bkj);
              //B(i, j) += A(i, k)*alpha_Bkj;
            }
          }
          for (idx_int k = n-1; k >= 0; --k) {
            B(j, k) = type_conv<Tb, Td>(ww[k]);
          }
        }
      }
    }
  }

  if(!hasworking) {
    delete [] ww;
  }

  #undef A
  #undef B
}

}
#endif
