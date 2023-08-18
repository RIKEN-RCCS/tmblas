//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _DOT_TMPL_HPP
# define _DOT_TMPL_HPP

namespace tmblas {

// =============================================================================
/// @return dot product, $x^H y$.
/// @see dotu for unconjugated version, $x^T y$.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x and y. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup dot

template<typename Ta, typename Tb, typename Td>
blas::scalar_type<Ta, Tb>  dot(
                     const idx_int n,  
                     Ta const *x, idx_int incx, 
                     Tb const *y, idx_int incy)
{
  typedef blas::scalar_type<Ta, Tb> scalar_t;
  typedef blas::real_type<Td> Tdreal;
  //  typedef blas::real_type<Ta> Tareal;    
  // check arguments
  blas_error_if( n < 0 );
  blas_error_if( incx == 0 );
  blas_error_if( incy == 0 );
  Td result(Tdreal(0));
  if(blas::is_complex<Ta>::value) {
    if (incx == 1 && incy == 1) {
      // unit stride
      for (idx_int i = 0; i < n; ++i) {
        mixedp_madd<Td, Ta, Tb>(result, conjg<Ta>(x[i]), y[i]);
      }
    }
    else {
      // non-unit stride
      idx_int ix = (incx > 0 ? 0 : (-n + 1)*incx);
      idx_int iy = (incy > 0 ? 0 : (-n + 1)*incy);
      for (idx_int i = 0; i < n; ++i) {
        mixedp_madd<Td, Ta, Tb>(result, conjg<Ta>(x[ix]), y[iy]);
        ix += incx;
        iy += incy;
      }
    }
  }
  else{
    if (incx == 1 && incy == 1) {
      // unit stride
      for (idx_int i = 0; i < n; ++i) {
        mixedp_madd<Td, Ta, Tb>(result, x[i], y[i]);
      }
    }
    else {
      // non-unit stride
      idx_int ix = (incx > 0 ? 0 : (-n + 1)*incx);
      idx_int iy = (incy > 0 ? 0 : (-n + 1)*incy);
      for (idx_int i = 0; i < n; ++i) {
        mixedp_madd<Td, Ta, Tb>(result, x[ix], y[iy]);
        ix += incx;
        iy += incy;
      }
    }
  }
  return type_conv<scalar_t, Td>(result);
}

}
#endif
