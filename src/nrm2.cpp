//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "nrm2_tmpl.hpp"

namespace tmblas {

template
half nrm2<half, half, float>(
     const idx_int n,
     half const *x,
     const idx_int incx);

template
half nrm2<std::complex<half>, std::complex<half>, std::complex<float> >(
     const idx_int n,
     std::complex<half> const *x,
     const idx_int incx);

template
double nrm2<double, double, quadruple>(
     const idx_int n,
     double const *x,
     const idx_int incx);

template
double nrm2<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     const idx_int n,
     std::complex<double> const *x,
     const idx_int incx);

#ifdef CBLAS_ROUTINES

template<>
float nrm2<float>(
     const idx_int n,
     float const *x,
     const idx_int incx)
{
  return cblas_snrm2((BLAS_INT) n, x, (BLAS_INT) incx);
}

template<>
double nrm2<double>(
     const idx_int n,
     double const *x,
     const idx_int incx)
{
  return cblas_dnrm2((BLAS_INT) n, x, (BLAS_INT) incx);
}

template<>
float nrm2<std::complex<float> >(
     const idx_int n,
     std::complex<float> const *x,
     const idx_int incx)
{
  return cblas_scnrm2((BLAS_INT) n, x, (BLAS_INT) incx);
}

template<>
double nrm2<std::complex<double> >(
     const idx_int n,
     std::complex<double> const *x,
     const idx_int incx)
{
  return cblas_dznrm2((BLAS_INT) n, x, (BLAS_INT) incx);
}
#else
template
float nrm2<float>(
     const idx_int n,
     float const *x,
     const idx_int incx);

template
double nrm2<double>(
     const idx_int n,
     double const *x,
     const idx_int incx);

template
float nrm2<std::complex<float> >(
     const idx_int n,
     std::complex<float> const *x,
     const idx_int incx);

template
double nrm2<std::complex<double> >(
     const idx_int n,
     std::complex<double> const *x,
     const idx_int incx);
#endif

template
half nrm2<half>(
     const idx_int n,
     half const *x,
     const idx_int incx);

template
half nrm2<std::complex<half> >(
     const idx_int n,
     std::complex<half> const *x,
     const idx_int incx);

template
quadruple nrm2<quadruple>(
     const idx_int n,
     quadruple const *x,
     const idx_int incx);

template
quadruple nrm2<std::complex<quadruple> >(
     const idx_int n,
     std::complex<quadruple> const *x,
     const idx_int incx);

template
octuple nrm2<octuple>(
     const idx_int n,
     octuple const *x,
     const idx_int incx);

template
octuple nrm2<std::complex<octuple> >(
     const idx_int n,
     std::complex<octuple> const *x,
     const idx_int incx);

}
