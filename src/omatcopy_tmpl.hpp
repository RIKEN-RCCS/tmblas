//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _OMATCOPY_TMPL_HPP
#define _OMATCOPY_TMPL_HPP

namespace tmblas {

template< typename Ta, typename Tb, typename Td >
void omatcopy(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    blas::scalar_type<Ta, Tb> alpha,
    Ta const *A, idx_int lda,
    Tb       *B, idx_int ldb)
{
  
  typedef blas::scalar_type<Ta, Tb> Tscalar;
  typedef blas::real_type<Tscalar> Tscalarreal;
  typedef blas::real_type<Tb> Tbreal;
  const Tscalar Tscalar_one(Tscalarreal(1)), Tscalar_zero(Tscalarreal(0));
  const Tb Tb_zero(Tbreal(0));

  #define A(i_, j_) A[ (i_) + (j_)*lda ]
  #define B(i_, j_) B[ (i_) + (j_)*ldb ]

  // check arguments
  blas_error_if( trans != blas::Op::NoTrans &&
                 trans != blas::Op::Trans &&
                 trans != blas::Op::ConjTrans &&
				 blas::op2char(trans) != 'R' && 
				 blas::op2char(trans) != 'r' );
  blas_error_if( rows < 0 );
  blas_error_if( cols < 0 );
  
  blas_error_if( lda < rows );
  blas_error_if( ldb < ((trans != blas::Op::NoTrans && blas::op2char(trans) != 'R' && blas::op2char(trans) != 'r') ? cols : rows) );

  if(alpha == Tscalar_zero) {
    for(idx_int j=0 ; j<cols ; ++j) {
      for(idx_int i=0 ; i<rows ; ++i) {
        B(i, j) = Tb_zero;
      }
    }
    return;
  }
  if(trans == blas::Op::NoTrans) {
    if(alpha == Tscalar_one) {
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          B(i, j) = type_conv<Tb, Ta>(A(i, j));
        }
      }
    }
    else {
      Td tmp;
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, A(i, j));
          B(i, j) = type_conv<Tb, Td>(tmp);
        }
      }
    }
  }
  else if(trans == blas::Op::Trans) {
    if(alpha == Tscalar_one) {
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          B(j, i) = type_conv<Tb, Ta>(A(i, j));
        }
      }
    }
    else {
      Td tmp;
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, A(i, j));
          B(j, i) = type_conv<Tb, Td>(tmp);
        }
      }
    }
  }
  else if(trans == blas::Op::ConjTrans) {
    if(alpha == Tscalar_one) {
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          B(j, i) = type_conv<Tb, Ta>(conjg<Ta>(A(i, j)));
        }
      }
    }
    else {
      Td tmp;
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, conjg<Ta>(A(i, j)));
          B(j, i) = type_conv<Tb, Td>(tmp);
        }
      }
    }
  }
  else {
    if(alpha == Tscalar_one) {
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          B(i, j) = type_conv<Tb, Ta>(conjg<Ta>(A(i, j)));
        }
      }
    }
    else {
      Td tmp;
      for(idx_int j=0 ; j<cols ; ++j) {
        for(idx_int i=0 ; i<rows ; ++i) {
          mixedp_mul<Td, Tscalar, Ta>(tmp, alpha, conjg<Ta>(A(i, j)));
          B(i, j) = type_conv<Tb, Td>(tmp);
        }
      }
    }
  }

  #undef A
  #undef B
}

}
#endif
