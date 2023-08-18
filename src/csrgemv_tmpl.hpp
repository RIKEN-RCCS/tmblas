//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRGEMV_TMPL_HPP
# define _CSRGEMV_TMPL_HPP

namespace tmblas{

// =============================================================================
/// General Sparse matrix-vector multiply:
/// \[
///     y = \alpha op(A) x + \beta y,
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
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] av, ai, aj
///     CSR matrix format with column size m.
///
/// @param[in] x
///     in an array of length m.
///
/// @param[in] beta
///     Scalar beta. If beta is zero, y need not be set on input.
///
/// @param[in, out] y
///     in an array of length m.
///
/// @param w
///     Work array of size column vector : m of A and vectors x and y/
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup csrgemv

template<typename Ta, typename Tb, typename Tc, typename Td>
void csrgemv(
     blas::Op trans,
     idx_int m,
     blas::scalar_type<Ta, Tb, Tc> const &alpha,     
     Ta const *av,
     idx_int const *ai,
     idx_int const *aj,
     Tb const *x,
     blas::scalar_type<Ta, Tb, Tc> const &beta,
     Tc *y,
     Td *w)
{
  bool hasworking = (!(w == nullptr) || (trans == blas::Op::NoTrans));
  Td *ww;

  if (!hasworking) {
    ww = new Td[m];
  }
  else {
    ww = w;
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
	for (idx_int i = 0; i < m; ++i) {
	  y[i] = Tczero;
	}
      }
      else {
	Td tmp;
	for(idx_int i = 0 ; i < m ; ++i) {
	  mixedp_mul<Td, Tc, scalar_t>(tmp, y[i], beta);
	  y[i] = type_conv<Tc, Td>(tmp);
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
	mixedp_mul<Td, Ta, Tb>(tmp, av[k], x[jj]);
      }
      for (idx_int k = ai[i] + 1; k < ai[i + 1]; ++k) {
	idx_int jj = aj[k];
	mixedp_madd<Td, Ta, Tb>(tmp, av[k], x[jj]);
      }
      mixedp_mul<Td, scalar_t, Td>(tmp, alpha, tmp);
      mixedp_madd<Td, scalar_t, Tc>(tmp, beta, y[i]);
      y[i] = type_conv<Tc, Td>(tmp);
    }
  }
  else if (trans == blas::Op::Trans) {
    for (idx_int i = 0; i < m; ++i) {
      ww[i] = Tdzero;
    }
    for (idx_int i = 0; i < m; ++i) {
      for (idx_int k = ai[i]; k < ai[i + 1]; ++k) {
	idx_int jj = aj[k];
	mixedp_madd<Td, Ta, Tb>(ww[jj], av[k], x[i]);
      }
    }
    for (idx_int i = 0; i < m; ++i) {
      mixedp_mul<Td, scalar_t, Td>(ww[i], alpha, ww[i]);
      mixedp_madd<Td, scalar_t, Tc>(ww[i], beta, y[i]);
      y[i] = type_conv<Tc, Td>(ww[i]);
    }
  }
  else {  // (trans == blas::Op::ConjTrans)
    for (idx_int i = 0; i < m; ++i) {
      ww[i] = Tdzero;
    }
    for (idx_int i = 0; i < m; ++i) {
      for (idx_int k = ai[i]; k < ai[i + 1]; ++k) {
	idx_int jj = aj[k];
	mixedp_madd<Td, Ta, Tb>(ww[jj], conjg<Ta>(av[k]), x[i]);
      }
    }
    for (idx_int i = 0; i < m; ++i) {
      mixedp_mul<Td, scalar_t, Td>(ww[i], alpha, ww[i]);      
      mixedp_madd<Td, scalar_t, Tc>(ww[i], beta, y[i]);
      y[i] = type_conv<Tc, Td>(ww[i]);
    }
  }
  if (!hasworking) {
    delete [] ww;
  }
}
} // namespace tmblas
#endif

