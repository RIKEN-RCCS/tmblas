//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "scal_tmpl.hpp"

namespace tmblas {

template
void scal<half, float>(
     const idx_int n,
     const half &alpha,
     half *x,
     const idx_int incx);
     
template
void scal<std::complex<half>, std::complex<float> >(
     const idx_int n,
     const std::complex<half> &alpha,
     std::complex<half> *x,
     const idx_int incx);

template
void scal<float, double>(
     const idx_int n,
     const float &alpha,
     float *x,
     const idx_int incx);
     
template
void scal<std::complex<float>, std::complex<double> >(
     const idx_int n,
     const std::complex<float> &alpha,
     std::complex<float> *x,
     const idx_int incx);

template
void scal<double, quadruple>(
     const idx_int n,
     const double &alpha,
     double *x,
     const idx_int incx);
     
template
void scal<std::complex<double>, std::complex<quadruple> >(
     const idx_int n,
     const std::complex<double> &alpha,
     std::complex<double> *x,
     const idx_int incx);

#ifdef CBLAS_ROUTINES

template<>
void scal<float>(
     const idx_int n,
     const float &alpha,
     float *x,
     const idx_int incx)
{
  cblas_sscal((BLAS_INT) n, alpha, x, (BLAS_INT) incx);
}

template<>
void scal<double>(
     const idx_int n,
     const double &alpha,
     double *x,
     const idx_int incx)
{
  cblas_dscal((BLAS_INT) n, alpha, x, (BLAS_INT) incx);
}

template<>
void scal<std::complex<float> >(
     const idx_int n,
     const std::complex<float> &alpha,
     std::complex<float> *x,
     const idx_int incx)
{
  cblas_cscal((BLAS_INT) n, (BLAS_VOID const *)&alpha, (BLAS_VOID *)x, (BLAS_INT) incx);
}

template<>
void scal<std::complex<double> >(
     const idx_int n,
     const std::complex<double> &alpha,
     std::complex<double> *x,
     const idx_int incx)
{
  cblas_zscal((BLAS_INT) n, (BLAS_VOID const *)&alpha, (BLAS_VOID *)x, (BLAS_INT) incx);
}
#else
template
void scal<float>(
     const idx_int n,
     const float &alpha,
     float *x,
     const idx_int incx);

template
void scal<double>(
     const idx_int n,
     const double &alpha,
     double *x,
     const idx_int incx);

template
void scal<std::complex<float> >(
     const idx_int n,
     const std::complex<float> &alpha,
     std::complex<float> *x,
     const idx_int incx);

template
void scal<std::complex<double> >(
     const idx_int n,
     const std::complex<double> &alpha,
     std::complex<double> *x,
     const idx_int incx);
#endif

template
void scal<half>(
     const idx_int n,
     const half &alpha,
     half *x,
     const idx_int incx);

template
void scal<std::complex<half> >(
     const idx_int n,
     const std::complex<half> &alpha,
     std::complex<half> *x,
     const idx_int incx);

template
void scal<quadruple>(
     const idx_int n,
     const quadruple &alpha,
     quadruple *x,
     const idx_int incx);

template
void scal<std::complex<quadruple> >(
     const idx_int n,
     const std::complex<quadruple> &alpha,
     std::complex<quadruple> *x,
     const idx_int incx);

template
void scal<octuple>(
     const idx_int n,
     const octuple &alpha,
     octuple *x,
     const idx_int incx);

template
void scal<std::complex<octuple> >(
     const idx_int n,
     const std::complex<octuple> &alpha,
     std::complex<octuple> *x,
     const idx_int incx);

}

