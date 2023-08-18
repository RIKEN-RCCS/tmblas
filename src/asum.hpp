//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _ASUM_HPP
#define _ASUM_HPP

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

template <typename T, typename Td = T>
blas::real_type<T>
asum(
    idx_int n,
    T const *x, idx_int incx );

#ifdef CBLAS_ROUTINES
template<>
float asum<float>(
		  idx_int n,  
		  float const *x, idx_int incx);

template<>
double asum<double>(
                     idx_int n,  
                     double const *x, idx_int incx);

template<>
float asum<std::complex<float> >(
                     idx_int n,  
                     std::complex<float> const *x, idx_int incx);

template<>
double asum<std::complex<double>>(
                     idx_int n,  
                     std::complex<double> const *x, idx_int incx);

#endif  
}  // namespace blas

#endif        //  #ifndef BLAS_ASUM_HH
