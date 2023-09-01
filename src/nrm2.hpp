//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _NRM2_HH
#define _NRM2_HH

namespace tmblas {

// =============================================================================
/// @return 2-norm of vector,
///     $|| x ||_2 = (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic implementation does not currently scale to avoid over- or underflow.
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
/// @ingroup nrm2

template< typename T, typename Tc = T, typename Td = T >
blas::real_type<Tc>
nrm2(
    const idx_int n,
    T const * x, const idx_int incx );

#ifdef CBLAS_ROUTINES

template<>
float nrm2<float>(
     const idx_int n,
     float const *x,
     const idx_int incx);

template<>
double nrm2<double>(
     const idx_int n,
     double const *x,
     const idx_int incx);

template<>
float nrm2<std::complex<float> >(
     const idx_int n,
     std::complex<float> const *x,
     const idx_int incx);

template<>
double nrm2<std::complex<double> >(
     const idx_int n,
     std::complex<double> const *x,
     const idx_int incx);

#endif

}

#endif

