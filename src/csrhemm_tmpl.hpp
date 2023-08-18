//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRHEMM_TMPL_HPP
# define _CSRHEMM_TMPL_HPP

namespace tmblas{

// =============================================================================
/// Hermitian Symmetric Sparse matrix-matrix multiply:
/// \[
///     y = \alpha A X + \beta Y,
/// \]
/// where alpha and beta are scalars, x and y are vectors,
/// and A is an m-by-m matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed from symmetry.
///     - Uplo::Lower: only the lower triangular part of A is referenced.
///     - Uplo::Upper: only the upper triangular part of A is referenced.
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
///     CSR matrix format with column size m and
///     only slower or upper part are stored.
///     For Lower triangular case, aj[k]==i with k = ai[i+1]-1 and then
///     existence of diagonal entries are supposed with value 0.
///     For Upper triangular case, aj[k]==i with k = ai[i] and then
///     existence of diagonal entries are supposed with value 0.
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
///     Work array of size column vector, m * n.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup csrhemm

//   only for  (layout == Layout::ColMajor)
template<typename Ta, typename Tb, typename Tc, typename Td>
void csrhemm(     
	     blas::Uplo uplo,
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
    ww = new Td[m * n];
  }

  typedef blas::scalar_type<Ta, Tb, Tc> scalar_t;
  typedef blas::real_type<Ta> Tareal;
  typedef blas::real_type<Tc> Tcreal;  

  const Tc Tczero(Tcreal(0));
  const Td Tdzero(Tcreal(0));  

  // check arguments
  blas_error_if( uplo != blas::Uplo::Upper &&
                 uplo != blas::Uplo::Lower);
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
  if (uplo == blas::Uplo::Lower) {
    for (idx_int i = 0; i < (m * n); ++i) {
      ww[i] = Tdzero;
    }
    for (idx_int i = 0; i < m; ++i) {      
      for (idx_int k = ai[i]; k < ai[i + 1] - 1; ++k) {
	idx_int jj = aj[k];
	Ta atmp(av[k]);
	for (idx_int j = 0; j < n; ++j) {		
	  mixedp_madd<Td, Ta, Tb>(ww[i + j * m], atmp, x[jj + j * ldx]);
	  mixedp_madd<Td, Ta, Tb>(ww[jj + j * m], conjg<Ta>(atmp), x[i + j * ldx]);	
	}
      }
      {
	idx_int k = ai[i + 1] - 1; // Lower part is stored
	idx_int jj = aj[k];
	Ta atmp(av[k]);	
	for (idx_int j = 0; j < n; ++j) {			
	  mixedp_madd<Td, Tareal, Tb>(ww[i + j * m], real<Tareal>(atmp), x[jj + j * ldx]);
	}
      }
    }
    for (idx_int i = 0; i < m; ++i) {
      for (idx_int j = 0; j < n; ++j) {			      
	mixedp_mul<Td, scalar_t, Td>(tmp, alpha, ww[i + j * m]);
	mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[i + j * ldy]);
	y[i + j * ldy] = type_conv<Tc, Td>(tmp);
      }
    }
  }
  else {
    for (idx_int i = 0; i < (m * n); ++i) {
      ww[i] = Tdzero;
    }
    for (idx_int i = 0; i < m; ++i) {
      {
	idx_int k = ai[i]; // Upper part is stored
	idx_int jj = aj[k];
	Ta atmp(av[k]);
	for (int j = 0; j < n; ++j) {
	  mixedp_madd<Td, Tareal, Tb>(ww[i + j * m], real<Tareal>(atmp), x[jj + j * ldx]);
	}
      }
      for (idx_int k = ai[i] + 1; k < ai[i + 1]; ++k) {
	idx_int jj = aj[k];
	Ta atmp(av[k]);
	for (int j = 0; j < n; ++j) {	
	  mixedp_madd<Td, Ta, Tb>(ww[i + j * m], atmp, x[jj + j * ldx]);
	  mixedp_madd<Td, Ta, Tb>(ww[jj + j * m], conjg<Ta>(atmp), x[i + j * ldx]);
	}
      }
    }
    for (idx_int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {	      
	mixedp_mul<Td, scalar_t, Td>(tmp, alpha, ww[i + j * m]);
	mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[i + j * ldy]);
	y[i + j * ldy] = type_conv<Tc, Td>(tmp);
      }
    }
  }
  if (!hasworking) {
    delete [] ww;
  }
}
} // namespace tmblas
#endif

