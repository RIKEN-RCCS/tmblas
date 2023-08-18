//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _IAMAX_HPP
# define _IAMAX_HPP

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

template<typename T>
idx_int iamax(
    idx_int n,
    T const *x, const idx_int incx );


#ifdef CBLAS_ROUTINES
template<>
idx_int iamax<float> (idx_int n, float const *x, const idx_int incx);

template<>
idx_int iamax<std::complex<float> >(idx_int n, std::complex<float> const *x, const idx_int incx);

template<>
idx_int iamax<double>(idx_int n, double const *x, const idx_int incx);

template<>
idx_int iamax<std::complex<double> >(idx_int n, std::complex<double> const *x, const idx_int incx);
#endif

}

#endif
