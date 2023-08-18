//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "axpy_tmpl.hpp"

namespace tmblas {

template
void axpy<half, half, float>(
     const idx_int n,
     half const &alpha, 
     half const *x, const idx_int incx, 
           half *y, const idx_int incy);

template
void axpy<std::complex<half>, std::complex<half>, std::complex<float> >(
     const idx_int n,
     std::complex<half> const &alpha, 
     std::complex<half> const *x, const idx_int incx, 
           std::complex<half> *y, const idx_int incy);

template
void axpy<float, float, double>(
     const idx_int n,
     float const &alpha, 
     float const *x, const idx_int incx, 
           float *y, const idx_int incy);

template
void axpy<std::complex<float>, std::complex<float>, std::complex<double> >(
     const idx_int n,
     std::complex<float> const &alpha, 
     std::complex<float> const *x, const idx_int incx, 
           std::complex<float> *y, const idx_int incy);

template
void axpy<double, double, quadruple>(
     const idx_int n,
     double const &alpha, 
     double const *x, const idx_int incx, 
           double *y, const idx_int incy);

template
void axpy<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     const idx_int n,
     std::complex<double> const &alpha, 
     std::complex<double> const *x, const idx_int incx, 
           std::complex<double> *y, const idx_int incy);

#ifdef CBLAS_ROUTINES
template<>
void axpy<float, float> (
     const idx_int n,
     float const &alpha, 
     float const *x, const idx_int incx, 
           float *y, const idx_int incy)
{
  cblas_saxpy(n, alpha, x, (BLAS_INT) incx, y, (BLAS_INT) incy);
}

template<>
void axpy<double, double> (
     const idx_int n,
     double const &alpha, 
     double const *x, const idx_int incx, 
           double *y, const idx_int incy)
{
  cblas_daxpy(n, alpha, x, (BLAS_INT) incx, y, (BLAS_INT) incy);
}

template<>
void axpy<std::complex<float>, std::complex<float> > (
     const idx_int n,
     std::complex<float> const &alpha, 
     std::complex<float> const *x, const idx_int incx, 
           std::complex<float> *y, const idx_int incy)
{
  cblas_caxpy(n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)x, (BLAS_INT) incx, (BLAS_VOID *)y, (BLAS_INT) incy);
}

template<>
void axpy<std::complex<double>, std::complex<double> > (
     const idx_int n,
     std::complex<double> const &alpha, 
     std::complex<double> const *x, const idx_int incx, 
           std::complex<double> *y, const idx_int incy)
{
  cblas_zaxpy(n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)x, (BLAS_INT) incx, (BLAS_VOID *)y, (BLAS_INT) incy);
}
#else
template
void axpy<float, float>(
     const idx_int n,
     float const &alpha, 
     float const *x, const idx_int incx, 
           float *y, const idx_int incy);

template
void axpy<double, double>(
     const idx_int n,
     double const &alpha, 
     double const *x, const idx_int incx, 
           double *y, const idx_int incy);

template
void axpy<std::complex<float>, std::complex<float> >(
     const idx_int n,
     std::complex<float> const &alpha, 
     std::complex<float> const *x, const idx_int incx, 
           std::complex<float> *y, const idx_int incy);

template
void axpy<std::complex<double>, std::complex<double> >(
     const idx_int n,
     std::complex<double> const &alpha, 
     std::complex<double> const *x, const idx_int incx, 
           std::complex<double> *y, const idx_int incy);
#endif

template
void axpy<half, half>(
     const idx_int n,
     half const &alpha, 
     half const *x, const idx_int incx, 
           half *y, const idx_int incy);

template
void axpy<std::complex<half>, std::complex<half> >(
     const idx_int n,
     std::complex<half> const &alpha, 
     std::complex<half> const *x, const idx_int incx, 
           std::complex<half> *y, const idx_int incy);

template
void axpy<quadruple, quadruple>(
     const idx_int n,
     quadruple const &alpha, 
     quadruple const *x, const idx_int incx, 
           quadruple *y, const idx_int incy);

template
void axpy<std::complex<quadruple>, std::complex<quadruple> >(
     const idx_int n,
     std::complex<quadruple> const &alpha, 
     std::complex<quadruple> const *x, const idx_int incx, 
           std::complex<quadruple> *y, const idx_int incy);

template
void axpy<octuple, octuple>(
     const idx_int n,
     octuple const &alpha, 
     octuple const *x, const idx_int incx, 
           octuple *y, const idx_int incy);

template
void axpy<std::complex<octuple>, std::complex<octuple> >(
     const idx_int n,
     std::complex<octuple> const &alpha, 
     std::complex<octuple> const *x, const idx_int incx, 
           std::complex<octuple> *y, const idx_int incy);

template
void axpy<float, half>(
     const idx_int n,
     float const &alpha, 
     float const *x, const idx_int incx, 
           half  *y, const idx_int incy);

template
void axpy<half, float>(
     const idx_int n,
     float const &alpha, 
     half  const *x, const idx_int incx, 
           float *y, const idx_int incy);

template
void axpy<std::complex<float>, std::complex<half> >(
     const idx_int n,
     std::complex<float> const &alpha, 
     std::complex<float> const *x, const idx_int incx, 
           std::complex<half>  *y, const idx_int incy);

template
void axpy<std::complex<half>, std::complex<float> >(
     const idx_int n,
     std::complex<float> const &alpha, 
     std::complex<half>  const *x, const idx_int incx, 
           std::complex<float> *y, const idx_int incy);

template
void axpy<double, float>(
     const idx_int n,
     double const &alpha, 
     double const *x, const idx_int incx, 
           float  *y, const idx_int incy);

template
void axpy<float, double>(
     const idx_int n,
     double const &alpha, 
     float  const *x, const idx_int incx, 
           double *y, const idx_int incy);

template
void axpy<std::complex<double>, std::complex<float> >(
     const idx_int n,
     std::complex<double> const &alpha, 
     std::complex<double> const *x, const idx_int incx, 
           std::complex<float>  *y, const idx_int incy);

template
void axpy<std::complex<float>, std::complex<double> >(
     const idx_int n,
     std::complex<double> const &alpha, 
     std::complex<float>  const *x, const idx_int incx, 
           std::complex<double> *y, const idx_int incy);

template
void axpy<quadruple,double>(
     const idx_int n,
     quadruple const &alpha, 
     quadruple const *x, const idx_int incx, 
           double    *y, const idx_int incy);

template
void axpy<double, quadruple>(
     const idx_int n,
     quadruple const &alpha, 
     double    const *x, const idx_int incx, 
           quadruple *y, const idx_int incy);

template
void axpy<std::complex<quadruple>, std::complex<double> >(
     const idx_int n,
     std::complex<quadruple> const &alpha, 
     std::complex<quadruple> const *x, const idx_int incx, 
           std::complex<double>    *y, const idx_int incy);

template
void axpy<std::complex<double>, std::complex<quadruple> >(
     const idx_int n,
     std::complex<quadruple> const &alpha, 
     std::complex<double> const *x, const idx_int incx, 
           std::complex<quadruple> *y, const idx_int incy);

}

