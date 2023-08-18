//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_IAMAX_HH
#define BLAS_IAMAX_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

template <typename T>
T abs1iamax( T const &x )
{
  T zero(0);
  if (x < zero)
    return -x;
  else
    return x;
}

template <typename T>
T abs1iamax( std::complex<T> const &x )
{
  return abs1iamax<T>(x.real()) + abs1iamax<T>(x.imag());  
}

// =============================================================================
/// @return Index of infinity-norm of vector, $|| x ||_{inf}$,
///     $\text{argmax}_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
/// Returns -1 if n = 0.
///
/// Generic implementation for arbitrary data types.
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
/// @ingroup iamax

template <typename T>
int64_t iamax(
    int64_t n,
    T const *x, int64_t incx )
{
    typedef real_type<T> real_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // todo: check NAN
    //    real_t result = -1;
    real_t result(-1);     //  29 Jun.2023 AS
    int64_t index = -1;
    if (incx == 1) {
      // unit stride
      for (int64_t i = 0; i < n; ++i) {
	real_t tmp = abs1iamax<real_type<T> >(x[i]);
	if (tmp > result) {
	  result = tmp;
	  index = i;
	}
      }
    }
    else {
      // non-unit stride
      int64_t ix = 0;
      for (int64_t i = 0; i < n; ++i) {
	real_t tmp = abs1iamax<real_type<T> >( x[ix] );
	if (tmp > result) {
	  result = tmp;
	  index = i;
	}
	ix += incx;
      }
    }
    return index;
}

}  // namespace blas

#endif        //  #ifndef BLAS_IAMAX_HH
