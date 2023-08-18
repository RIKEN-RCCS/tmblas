//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _AXPY_HPP
# define _AXPY_HPP

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

template<typename Ta, typename Tb, typename Td = blas::scalar_type<Ta, Tb> > 
void axpy(
     const idx_int n, 
     blas::scalar_type<Ta, Tb> const &alpha, 
     Ta const * x, const int incx, 
     Tb *y, const int incy);

#ifdef CBLAS_ROUTINES
template<>
void axpy<float, float> (
     const idx_int n,
     float const &alpha, 
     float const *x, const idx_int incx, 
           float *y, const idx_int incy);

template<>
void axpy<double, double> (
     const idx_int n,
     double const &alpha, 
     double const *x, const idx_int incx, 
           double *y, const idx_int incy);

template<>
void axpy<std::complex<float>, std::complex<float> > (
     const idx_int n,
     std::complex<float> const &alpha, 
     std::complex<float> const *x, const idx_int incx, 
           std::complex<float> *y, const idx_int incy);

template<>
void axpy<std::complex<double>, std::complex<double> > (
     const idx_int n,
     std::complex<double> const &alpha, 
     std::complex<double> const *x, const idx_int incx, 
           std::complex<double> *y, const idx_int incy);
#endif

}

#endif

