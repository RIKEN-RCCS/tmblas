//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _DOTU_HPP
# define _DOTU_HPP

namespace tmblas {

// =============================================================================
/// @return unconjugated dot product, $x^T y$.
/// @see dotu for conjugated version, $x^H y$.
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

template<typename Ta, typename Tb, typename Tc = blas::scalar_type<Ta, Tb>, typename Td = Tc > 
Tc dotu(
                     const idx_int n,  
                     Ta const *x, int incx, 
                     Tb const *y, int incy);

#ifdef CBLAS_ROUTINES
template<>
std::complex<float> dotu<std::complex<float>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy);

template<>
std::complex<double> dotu<std::complex<double>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy);
#endif

}


#endif
