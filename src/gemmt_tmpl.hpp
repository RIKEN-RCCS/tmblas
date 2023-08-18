//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _GEMMT_TMPL_HPP
# define _GEMMT_TMPL_HPP

namespace tmblas {

// =============================================================================
/// General matrix-matrix multiply:
/// \[
///     C = \alpha op(A) \times op(B) + \beta C,
/// \]
/// where $op(X)$ is one of
///     $op(X) = X$,
///     $op(X) = X^T$, or
///     $op(X) = X^H$,
/// alpha and beta are scalars, and A, B, and C are matrices, with
/// $op(A)$ an m-by-k matrix, $op(B)$ a k-by-n matrix, and C an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] transA
///     The operation $op(A)$ to be used:
///     - Op::NoTrans:   $op(A) = A$.
///     - Op::Trans:     $op(A) = A^T$.
///     - Op::ConjTrans: $op(A) = A^H$.
///
/// @param[in] transB
///     The operation $op(B)$ to be used:
///     - Op::NoTrans:   $op(B) = B$.
///     - Op::Trans:     $op(B) = B^T$.
///     - Op::ConjTrans: $op(B) = B^H$.
///
/// @param[in] m
///     Number of rows of the matrix C and $op(A)$. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix C and $op(B)$. n >= 0.
///
/// @param[in] k
///     Number of columns of $op(A)$ and rows of $op(B)$. k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If transA = NoTrans:
///       the m-by-k matrix A, stored in an lda-by-k array [RowMajor: m-by-lda].
///     - Otherwise:
///       the k-by-m matrix A, stored in an lda-by-m array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If transA = NoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
///     - Otherwise:           lda >= max(1, k) [RowMajor: lda >= max(1, m)].
///
/// @param[in] B
///     - If transB = NoTrans:
///       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
///     - Otherwise:
///       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If transB = NoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
///     - Otherwise:           ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The n-by-n matrix C, stored in an ldc-by-n array [RowMajor: n-by-ldc].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
///
/// @ingroup gemm

//template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> >
//void gemm(blas::Op transA,
//           blas::Op transB,
//          idx_int m, idx_int n, idx_int k,
//          blas::scalar_type<Ta, Tb, Tc> const &alpha,
//          Ta const *A, int lda,
//          Tb const *B, int ldb,
//          blas::scalar_type<Ta, Tb, Tc> const &beta,
//          Tc *C, idx_int ldc);

template<typename Ta, typename Tb, typename Tc, typename Td >
inline void gemmt(blas::Uplo uplo,
		  blas::Op transA,
		  blas::Op transB,
		  idx_int n, idx_int k,
		  blas::scalar_type<Ta, Tb, Tc> const &alpha,
		  Ta const *A, idx_int lda,
		  Tb const *B, idx_int ldb,
		  blas::scalar_type<Ta, Tb, Tc> const &beta,
		  Tc *C, idx_int ldc) {

  typedef blas::scalar_type<Ta, Tb, Tc> Tscalar;
  //  typedef blas::real_type<Tscalar> Tscalarreal;
  typedef blas::real_type<Tc> Tcreal;
  typedef blas::real_type<Td> Tdreal;  
  //  const Tscalar Tscalar_zero(Tscalarreal(0)), Tscalar_one(Tscalarreal(1));
  const Tc Tc_zero(Tcreal(0));
  const Td Td_zero(Tdreal(0));

  #define A(i_, j_) A[ (i_) + (j_)*lda ]
  #define B(i_, j_) B[ (i_) + (j_)*ldb ]
  #define C(i_, j_) C[ (i_) + (j_)*ldc ]
  // check arguments
  blas_error_if( transA != blas::Op::NoTrans &&
                 transA != blas::Op::Trans &&
                 transA != blas::Op::ConjTrans );
  blas_error_if( transB != blas::Op::NoTrans &&
                 transB != blas::Op::Trans &&
                 transB != blas::Op::ConjTrans );
  blas_error_if( n < 0 );
  blas_error_if( k < 0 );

  blas_error_if( lda < ((transA != blas::Op::NoTrans) ? k : n) );
  blas_error_if( ldb < ((transB != blas::Op::NoTrans) ? n : k) );
  blas_error_if( ldc < n );

  // quick return
  if (n == 0 || k == 0) {
      return;
  }

  if(mixedp_eq<Tscalar, int>(alpha, 0)) {
    if(mixedp_eq<Tscalar, int>(beta, 0)) {
      if (uplo != blas::Uplo::Upper) {      
	for (idx_int j = 0 ; j < n ; ++j) {
	  for (idx_int i = 0 ; i <=j ; ++i) {
	    C(i, j) = Tc_zero;
	  }
	}
      }
      else if (uplo != blas::Uplo::Lower) {      
	for (idx_int j = 0 ; j < n ; ++j) {
	  for (idx_int i = j ; i < n ; ++i) {
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
    else {
      Td tmp;
      if (uplo != blas::Uplo::Upper) {            
	for (idx_int j = 0 ; j < n ; ++j) {
	  for (idx_int i = 0 ; i <=j ; ++i) {
	    mixedp_mul<Td, Tscalar, Tc>(tmp, beta, C(i, j));
	    C(i, j) = type_conv<Tc, Td>(tmp);
	  }
	}
      }
      else if (uplo != blas::Uplo::Lower) {
	for (idx_int j = 0 ; j < n ; ++j) {
	  for (idx_int i = 0 ; i <=j ; ++i) {
	    mixedp_mul<Td, Tscalar, Tc>(tmp, beta, C(i, j));
	    C(i, j) = type_conv<Tc, Td>(tmp);
	  }
	}
      }
      else {
	for (idx_int j = 0 ; j < n ; ++j) {
	  for (idx_int i = 0 ; i < n ; ++i) {
	    mixedp_mul<Td, Tscalar, Tc>(tmp, beta, C(i, j));
	    C(i, j) = type_conv<Tc, Td>(tmp);
	  }
	}
      }
    }
    return;
  }

  if (transA == blas::Op::NoTrans) {
    if (transB == blas::Op::NoTrans) {
  // scalar temporary value for higher precision arithmetic
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {      
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    }
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(i, l), B(l, j));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    }
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(i, l), B(l, j));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
	
      }
    }
    else if (transB == blas::Op::Trans) {
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(i, l), B(j, l));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(i, l), B(j, l));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
    }
    else { // transB == Op::ConjTrans
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {      
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <=j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(i, l), conjg<Tb>(B(j, l)));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(i, l), conjg<Tb>(B(j, l)));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
    }
  }
  else if (transA == blas::Op::Trans) {
    if (transB == blas::Op::NoTrans) {
  // scalar temporary value for higher precision arithmetic
        // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {      
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(l, i), B(l, j));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(l, i), B(l, j));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
    }
    else if (transB == blas::Op::Trans) {
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {            
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(l, i), B(j, l));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(l, i), B(j, l));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
	
      } 
    }
    else { // transB == Op::ConjTrans
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {     
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(l, i), conjg<Tb>(B(j, l)));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, A(l, i), conjg<Tb>(B(j, l)));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
    }
  }
  else { // A conjg
    if (transB == blas::Op::NoTrans) {
  // scalar temporary value for higher precision arithmetic
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {           
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, conjg<Ta>(A(l, i)), B(l, j));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, conjg<Ta>(A(l, i)), B(l, j));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
    }
    else if (transB == blas::Op::Trans) {
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {      
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, conjg<Ta>(A(l, i)), B(j, l));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, conjg<Ta>(A(l, i)), B(j, l));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
    }
    else { // transB == Op::ConjTrans
      // C[i, j] = \beta C[i, j]; C[i, j] += \alpha \sum_{l} (A[i, l] * B[l, j])
      Td work0, work1;
      if (uplo != blas::Uplo::Lower) {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = 0; i <= j; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, conjg<Ta>(A(l, i)), conjg<Tb>(B(j, l)));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
	}
      }
      else {
	for (idx_int j = 0; j < n; ++j) {
	  for (idx_int i = j; i < n; ++i) {
	    if (mixedp_eq<Tscalar, int>(beta, 1)) {
	      work0 = type_conv<Td, Tc>(C(i,j));
	    } 
	    else if (mixedp_eq<Tscalar, int>(beta, 0)) {
	      work0 = Td_zero;
	    }
	    else {
	      mixedp_mul<Td, Tc, Tscalar>(work0, C(i, j), beta);
	    }
	    work1 = Td_zero;
	    for (idx_int l = 0; l < k; ++l) {
	      mixedp_madd<Td, Ta, Tb>(work1, conjg<Ta>(A(l, i)), conjg<Tb>(B(j, l)));
	    }
	    mixedp_madd<Td, Tscalar, Td>(work0, alpha, work1);
	    C(i, j) = type_conv<Tc, Td>(work0);
	  }
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
  
  #undef A
  #undef B
  #undef C
}

}


#endif

