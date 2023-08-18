//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "iamax_tmpl.hpp"

namespace tmblas {

template
idx_int iamax<half>(idx_int n, half const *x, const idx_int incx);

template
idx_int iamax<std::complex<half> >(idx_int n, std::complex<half> const *x, const idx_int incx);

#ifdef CBLAS_ROUTINES
template<>
idx_int iamax<float> (idx_int n, float const *x, const idx_int incx)
{
  return cblas_isamax((BLAS_INT) n, x, (BLAS_INT) incx);
}

template<>
idx_int iamax<std::complex<float> >(idx_int n, std::complex<float> const *x, const idx_int incx)
{
  return cblas_icamax((BLAS_INT) n, x, (BLAS_INT) incx);
}

template<>
idx_int iamax<double>(idx_int n, double const *x, const idx_int incx)
{
  return cblas_idamax((BLAS_INT) n, x, (BLAS_INT) incx);
}

template<>
idx_int iamax<std::complex<double> >(idx_int n, std::complex<double> const *x, const idx_int incx)
{
  return cblas_izamax((BLAS_INT) n, x, (BLAS_INT) incx);
}
#else
template
idx_int iamax<float>(idx_int n, float const *x, const idx_int incx);

template
idx_int iamax<std::complex<float> >(idx_int n, std::complex<float> const *x, const idx_int incx);

template
idx_int iamax<double>(idx_int n, double const *x, const idx_int incx);

template
idx_int iamax<std::complex<double> >(idx_int n, std::complex<double> const *x, const idx_int incx);
#endif
template
idx_int iamax<quadruple>(idx_int n, quadruple const *x, const idx_int incx);

template
idx_int iamax<octuple>(idx_int n, octuple const *x, const idx_int incx);

template
idx_int iamax<std::complex<quadruple> >(idx_int n, std::complex<quadruple> const *x, const idx_int incx);

template
idx_int iamax<std::complex<octuple> >(idx_int n, std::complex<octuple> const *x, const idx_int incx);
}

