//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _IAMAX_TMPL_HPP
# define _IAMAX_TMPL_HPP

namespace tmblas {

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

template< typename T >
idx_int iamax(
    idx_int n,
    T const *x, const idx_int incx )
{
    typedef blas::real_type<T> real_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // todo: check NAN
    real_t result(-1);
    idx_int index = -1;
    if (incx == 1) {
        // unit stride
        for (idx_int i = 0; i < n; ++i) {
	  real_t tmp = abs1<real_t>( const_cast<T &>(x[i]) );
	  if (mixedp_gt<real_t, real_t>(tmp, result)) {
                result = tmp;
                index = i;
            }
        }
    }
    else {
        // non-unit stride
        idx_int ix = 0;
        for (idx_int i = 0; i < n; ++i) {
	  real_t tmp = abs1<real_t>( const_cast<T &>(x[ix]) );
	  if (mixedp_gt<real_t, real_t>(tmp, result)) {
                result = tmp;
                index = i;
            }
            ix += incx;
        }
    }
    return index;
}

}
#endif
