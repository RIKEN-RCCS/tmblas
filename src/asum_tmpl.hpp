//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _ASUM_TMPL_HPP
#define _ASUM_TMPL_HPP

namespace tmblas {

// =============================================================================
/// @return 1-norm of vector,
///     $|| Re(x) ||_1 + || Im(x) ||_1
///         = \sum_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
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
/// @ingroup asum

template <typename T, typename Td>
blas::real_type<T>
asum(
    idx_int n,
    T const *x, idx_int incx )
{
  typedef blas::real_type<T> real_t;
  typedef blas::real_type<Td> Tdreal;
    
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    Tdreal result(0); 
    if (incx == 1) {
        // unit stride
        for (idx_int i = 0; i < n; ++i) {
	  mixedp_add<Tdreal, Tdreal, real_t>(result, result, abs1<real_t>( const_cast<T &>(x[i]) ));
        }
    }
    else {
        // non-unit stride
        idx_int ix = 0;
        for (idx_int i = 0; i < n; ++i) {
	  mixedp_add<Tdreal, Tdreal, real_t>(result, result, abs1<real_t>( const_cast<T &>(x[ix]) ));	  
            ix += incx;
        }
    }
    return type_conv<real_t, Tdreal>(result);
}

}  // namespace blas

#endif        //  #ifndef BLAS_ASUM_HH
