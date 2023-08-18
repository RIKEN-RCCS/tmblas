//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRGEMM_TMPL_HPP
# define _CSRGEMM_TMPL_HPP

namespace tmblas{

// =============================================================================
/// General Sparse matrix-matrix multiply:
/// \[
///     y = \alpha op(A) X + \beta Y,
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
///     Number of columns of the matrix X and Y. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and X are not accessed.
///
/// @param[in] av, ai, aj
///     CSR matrix format with column size m.
///
/// @param[in] x
///     in an array of length m x n.
///
/// @param[in] ldx
///     Leading dimension of X.
///
/// @param[in] beta
///     Scalar beta. If beta is zero, y need not be set on input.
///
/// @param[in, out] y
///     in an array of length m x n.
///
/// @param[in] ldy
///     Leading dimension of Y.
///
/// @param w
///     Work array of size column vector : m for NoTrans, m * n otherwise.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup csrgemm

//   only for  (layout == Layout::ColMajor)
template<typename Ta, typename Tb, typename Tc, typename Td>
void csrgemm(
     blas::Op trans,
     idx_int m,
     idx_int n,     
     blas::scalar_type<Ta, Tb, Tc> const &alpha,     
     Ta const *av,
     idx_int const *ai,
     idx_int const *aj,
     Tb const *x,
     idx_int ldx,          
     blas::scalar_type<Ta, Tb, Tc> const &beta,
     Tc *y,
     idx_int ldy,               
     Td *w)
{
  bool hasworking = (!(w == nullptr));
  Td *ww;

  if (hasworking){
    ww = w;
  }
  else {
    int worksize;
    if (trans == blas::Op::NoTrans) {
      worksize = n;
    }
    else {
      worksize = m * n;
    }
    ww = new Td[worksize];
  }
  typedef blas::scalar_type<Ta, Tb, Tc> scalar_t;
  //  typedef blas::real_type<scalar_t> scalar_treal;  
  typedef blas::real_type<Tc> Tcreal;
  typedef blas::real_type<Td> Tdreal;  

  const Tc Tczero(Tcreal(0));
  const Td Tdzero(Tdreal(0));  

  // check arguments
  blas_error_if( trans != blas::Op::NoTrans &&
                 trans != blas::Op::Trans &&
                 trans != blas::Op::ConjTrans );
  blas_error_if( m < 0 );
  
  // quick return
  if (m == 0 || (mixedp_eq<scalar_t, int>(alpha, 0) &&
		 mixedp_eq<scalar_t, int>(beta, 1)))
    return;

    // ----------
    // form y = beta*y
  if (mixedp_eq<scalar_t, int>(alpha, 0)) {
    if (!mixedp_eq<scalar_t, int>(beta, 1)) {
      if (mixedp_eq<scalar_t, int>(beta, 0)) {
	for (idx_int j = 0; j < n; ++j) {	
	  for (idx_int i = 0; i < m; ++i) {
	    y[i + j * ldy] = Tczero;
	  }
	}
      }
      else {
	Td tmp;
	for (idx_int j = 0; j < n; ++j) {		
	  for(idx_int i = 0 ; i < m ; ++i) {
	    mixedp_mul<Td, Tc, scalar_t>(tmp, y[i], beta);
	    y[i + j * ldy] = type_conv<Tc, Td>(tmp);
	  }
	}
      }
    }
    return;
  }
  Td tmp;
  if (trans == blas::Op::NoTrans) {
    for (idx_int i = 0; i < m; ++i) {    
      { 
	idx_int k = ai[i];
	idx_int jj = aj[k];
	Ta atmp(av[k]);
	for (idx_int j = 0; j < n; ++j) {	
	  mixedp_mul<Td, Ta, Tb>(ww[j], atmp, x[jj + j * ldx]);
	} // loop : j among colums
      }
      for (idx_int k = ai[i] + 1; k < ai[i + 1]; ++k) {
	idx_int jj = aj[k];
	Ta atmp(av[k]);
	for (idx_int j = 0; j < n; ++j) {	
	  mixedp_madd<Td, Ta, Tb>(ww[j], atmp, x[jj + j * ldx]);
	} // loop : j among colusm
      }
      for (idx_int j = 0; j < n; ++j) {	      
	mixedp_mul<Td, scalar_t, Td>(tmp, alpha, ww[j]);
	mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[i + j * ldy]);
	y[i + j * ldy] = type_conv<Tc, Td>(tmp);
      }  // loop : j among colums
    } 
  }
  else if (trans == blas::Op::Trans) {
    for (idx_int i = 0; i < (m * n); ++i) {
      ww[i] = Tdzero;
    }
    for (idx_int i = 0; i < m; ++i) {
      for (idx_int k = ai[i]; k < ai[i + 1]; ++k) {
	idx_int jj = aj[k];
	Ta atmp(av[k]);	  
	for (idx_int j = 0; j < n; ++j) {		  
	  mixedp_madd<Td, Ta, Tb>(ww[jj + j * m], atmp, x[i + j * ldx]);
	}
      }
    }  // loop : i
    for (idx_int i = 0; i < m; ++i) {
      for (idx_int j = 0; j < n; ++j) {
	mixedp_mul<Td, scalar_t, Td>(tmp, alpha, ww[i + j * m]);
	mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[i + j * ldy]);
	y[i + j * ldy] = type_conv<Tc, Td>(tmp);
      }
    }  // loop : i
  }
  else { // (trans == blas::Op::ConjTrans)
    for (idx_int i = 0; i < (m * n); ++i) {
      ww[i] = Tdzero;
    }
    for (idx_int i = 0; i < m; ++i) {
      for (idx_int k = ai[i]; k < ai[i + 1]; ++k) {
	idx_int jj = aj[k];
	Ta atmp(conjg<Ta>(av[k]));	  
	for (idx_int j = 0; j < n; ++j) {		  
	  mixedp_madd<Td, Ta, Tb>(ww[jj + j * m], atmp, x[i + j * ldx]);
	}
      }
    }  // loop : i
    for (idx_int i = 0; i < m; ++i) {
      for (idx_int j = 0; j < n; ++j) {
	mixedp_mul<Td, scalar_t, Td>(tmp, alpha, ww[i + j * m]);
	mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[i + j * ldy]);
	y[i + j * ldy] = type_conv<Tc, Td>(tmp);
      }
    }  // loop : i
  }
  if (!hasworking) {
    delete [] ww;
  }
}
} // namespace tmblas
#endif

