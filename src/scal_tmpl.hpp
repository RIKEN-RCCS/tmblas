//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SCAL_TMPL_HH
#define _SCAL_TMPL_HH

namespace tmblas {

// =============================================================================
/// Scale vector by constant, $x = \alpha x$.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup scal

template<typename T, typename Td >
void scal(
    const idx_int n,
    T const &alpha,
    T *x, const idx_int incx)
{
  // check arguments
  blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
  blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

  Td tmp;
  if (incx == 1) {
    // unit stride
    for (idx_int i = 0; i < n; ++i) {
      mixedp_mul<Td, T, T>(tmp, x[i], alpha);
      x[i] = type_conv<T, Td>(tmp);
    }
  }
  else {
    // non-unit stride
    for (idx_int i = 0; i < n*incx; i += incx) {
      mixedp_mul<Td, T, T>(tmp, x[i], alpha);
      x[i] = type_conv<T, Td>(tmp);
    }
  }
}

}
#endif
