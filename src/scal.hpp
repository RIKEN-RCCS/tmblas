//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _SCAL_HH
#define _SCAL_HH

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

template<typename T, typename Td = T >
void scal(
    const idx_int n,
    T const &alpha,
    T *x, const idx_int incx);

#ifdef CBLAS_ROUTINES

template<>
void scal<float>(
     const idx_int n,
     const float &alpha,
     float *x,
     const idx_int incx);

template<>
void scal<double>(
     const idx_int n,
     const double &alpha,
     double *x,
     const idx_int incx);

template<>
void scal<std::complex<float> >(
     const idx_int n,
     const std::complex<float> &alpha,
     std::complex<float> *x,
     const idx_int incx);

template<>
void scal<std::complex<double> >(
     const idx_int n,
     const std::complex<double> &alpha,
     std::complex<double> *x,
     const idx_int incx);

#endif

}

#endif
