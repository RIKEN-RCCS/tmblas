//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
#ifndef _AXPY_TMPL_HPP
# define _AXPY_TMPL_HPP

namespace tmblas {

// =============================================================================
/// Add scaled vector, $y = \alpha x + y$.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x and y. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, y is not updated.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in, out] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup axpy

template<typename Ta, typename Tb, typename Td>
void axpy(
     const idx_int n, 
     blas::scalar_type<Ta, Tb> const &alpha, 
     Ta const *x, const idx_int incx, 
     Tb *y, const idx_int incy)
{
  typedef blas::scalar_type<Ta, Tb> Tscalar;
  //  typedef blas::real_type<Tscalar> Tscalarreal;  
  // check arguments
  blas_error_if( n < 0 );
  blas_error_if( incx == 0 );
  blas_error_if( incy == 0 );

  // quick return
  if (mixedp_eq<Tscalar, int>(alpha, 0)) {
      return;
  }

  if (incx == 1 && incy == 1) {
      // unit stride
    for (idx_int i = 0; i < n; ++i) {
        //y[i] += alpha*x[i];
      Td work = type_conv<Td,Tb>(y[i]);
      mixedp_madd<Td, Tscalar, Ta>(work,alpha,x[i]);
      y[i] = type_conv<Tb,Td>(work);
    }
  }
  else {
    // non-unit stride
    idx_int ix = (incx > 0 ? 0 : (-n + 1)*incx);
    idx_int iy = (incy > 0 ? 0 : (-n + 1)*incy);
    for (int64_t i = 0; i < n; ++i) {
      //y[iy] += alpha * x[ix];
      Td work = type_conv<Td,Tb>(y[iy]);
      mixedp_madd<Td, Tscalar, Ta>(work,alpha,x[ix]);
      y[iy] = type_conv<Tb,Td>(work);
      ix += incx;
      iy += incy;
    }
  }
}

}
#endif
