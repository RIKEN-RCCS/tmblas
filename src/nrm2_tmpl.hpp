//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _NRM2_TMPL_HH
#define _NRM2_TMPL_HH

namespace tmblas {

// =============================================================================
/// @return 2-norm of vector,
///     $|| x ||_2 = (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic implementation does not currently scale to avoid over- or underflow.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup nrm2

template< typename T, typename Td>
blas::real_type<T>
nrm2(
    const idx_int n,
    T const * x, const idx_int incx ) {
    typedef blas::real_type<T> real_t;
    typedef blas::real_type<Td> reald_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // todo: scale to avoid overflow & underflow
    reald_t result(0);
    if(blas::is_complex<T>::value) {
      //reald_t tmpr,tmpi;
      if (incx == 1) {
          // unit stride
          for (idx_int i = 0; i < n; ++i) {
	    real_t rpart = real<real_t>(const_cast<T &>(x[i]));
	    real_t ipart = imag<real_t>(const_cast<T &>(x[i]));
	    mixedp_madd<reald_t, real_t, real_t>(result, rpart, rpart);
	    mixedp_madd<reald_t, real_t, real_t>(result, ipart, ipart);
	    //result += tmpr+tmpi;
          }
      }
      else {
          // non-unit stride
          idx_int ix = 0;
          for (idx_int i = 0; i < n; ++i) {
	    real_t rpart = real<real_t>(const_cast<T &>(x[ix]));
	    real_t ipart = imag<real_t>(const_cast<T &>(x[ix]));
              mixedp_madd<reald_t, real_t, real_t>(result, rpart, rpart);
              mixedp_madd<reald_t, real_t, real_t>(result, ipart, ipart);
              //result += real(x[ix]) * real(x[ix]) + imag(x[ix]) * imag(x[ix]);
              ix += incx;
          }
      }
    }
    else{
      //reald_t tmpr,tmpi;
      if (incx == 1) {
          // unit stride
	for (idx_int i = 0; i < n; ++i) {
	  real_t rpart = real<real_t>(const_cast<T &>(x[i]));	  
              mixedp_madd<reald_t, real_t, real_t>(result, rpart, rpart);
              //result += tmpr+tmpi;
          }
      }
      else {
          // non-unit stride
          idx_int ix = 0;
          for (idx_int i = 0; i < n; ++i) {
	    real_t rpart = real<real_t>(const_cast<T &>(x[ix]));    
              mixedp_madd<reald_t, real_t, real_t>(result, rpart, rpart);
              //result += real(x[ix]) * real(x[ix]) + imag(x[ix]) * imag(x[ix]);
              ix += incx;
          }
      }
    }
    result = sqrt1<reald_t>(result);
    //return std::sqrt( result );
    return type_conv<real_t, reald_t>(result);
}

}
#endif
