//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "dot_tmpl.hpp"

namespace tmblas {

template 
half dot<half, half, float>(
                     const idx_int n,  
                     half const *x, idx_int incx, 
                     half const *y, idx_int incy);

template 
std::complex<half> dot<std::complex<half>, std::complex<half>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<half> const *x, idx_int incx, 
                     std::complex<half> const *y, idx_int incy);

template 
float dot<float, float, double>(
                     const idx_int n,  
                     float const *x, idx_int incx, 
                     float const *y, idx_int incy);

template 
std::complex<float> dot<std::complex<float>, std::complex<float>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy);

template 
double dot<double, double, quadruple>(
                     const idx_int n,  
                     double const *x, idx_int incx, 
                     double const *y, idx_int incy);

template 
std::complex<double> dot<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy);

#ifdef CBLAS_ROUTINES
template<>
float dot<float, float>(
                     const idx_int n,  
                     float const *x, idx_int incx, 
                     float const *y, idx_int incy) 
{
  return cblas_sdot(n, x, (BLAS_INT) incx, y, (BLAS_INT) incy);
}

template<>
double dot<double, double>(
                     const idx_int n,  
                     double const *x, idx_int incx, 
                     double const *y, idx_int incy) 
{
  return cblas_ddot(n, x, (BLAS_INT) incx, y, (BLAS_INT) incy);
}


template<>
std::complex<float> dot<std::complex<float>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy) 
{
  std::complex<float> result;
  cblas_cdotc_sub(n, (BLAS_VOID const *)x, (BLAS_INT) incx, (BLAS_VOID const *)y, (BLAS_INT) incy, (BLAS_VOID *)&result);
  return result;
}

template<>
std::complex<double> dot<std::complex<double>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy)
{
  std::complex<double> result;
  cblas_zdotc_sub(n, (BLAS_VOID const *)x, (BLAS_INT) incx, (BLAS_VOID const *)y, (BLAS_INT) incy, (BLAS_VOID *)&result);
  return result;
}

#else
template
float dot<float, float>(
                     const idx_int n,  
                     float const *x, idx_int incx, 
                     float const *y, idx_int incy);

template
double dot<double, double>(
                     const idx_int n,  
                     double const *x, idx_int incx, 
                     double const *y, idx_int incy);

template
std::complex<float> dot<std::complex<float>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy);

template
std::complex<double> dot<std::complex<double>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy);
#endif

template
half dot<half, half>(
                     const idx_int n,  
                     half const *x, idx_int incx, 
                     half const *y, idx_int incy);

template
quadruple dot<quadruple, quadruple>(
                     const idx_int n,  
                     quadruple const *x, idx_int incx, 
                     quadruple const *y, idx_int incy);

template
octuple dot<octuple, octuple>(
                     const idx_int n,  
                     octuple const *x, idx_int incx, 
                     octuple const *y, idx_int incy);

template
std::complex<half> dot<std::complex<half>, std::complex<half> >(
                     const idx_int n,  
                     std::complex<half> const *x, idx_int incx, 
                     std::complex<half> const *y, idx_int incy);

template
std::complex<quadruple> dot<std::complex<quadruple>, std::complex<quadruple> >(
                     const idx_int n,  
                     std::complex<quadruple> const *x, idx_int incx, 
                     std::complex<quadruple> const *y, idx_int incy);
template
std::complex<octuple> dot<std::complex<octuple>, std::complex<octuple> >(
                     const idx_int n,  
                     std::complex<octuple> const *x, idx_int incx, 
                     std::complex<octuple> const *y, idx_int incy);

template
float dot<float, half>(
                     const idx_int n,  
                     float const *x, idx_int incx, 
                     half  const *y, idx_int incy);

template
float dot<half, float>(
                     const idx_int n,  
                     half  const *x, idx_int incx, 
                     float const *y, idx_int incy);

template
std::complex<float> dot<std::complex<float>, std::complex<half> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<half>  const *y, idx_int incy);
template
std::complex<float> dot<std::complex<half>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<half>  const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy);

template
double dot<double, float>(
                     const idx_int n,  
                     double const *x, idx_int incx, 
                     float  const *y, idx_int incy);

template
double dot<float, double>(
                     const idx_int n,  
                     float  const *x, idx_int incx, 
                     double const *y, idx_int incy);

template
std::complex<double> dot<std::complex<double>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<float>  const *y, idx_int incy);

template
std::complex<double> dot<std::complex<float>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<float>  const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy);

template
quadruple dot<quadruple, double>(
                     const idx_int n,  
                     quadruple const *x, idx_int incx, 
                     double    const *y, idx_int incy);

template
quadruple dot<double, quadruple>(
                     const idx_int n,  
                     double    const *x, idx_int incx, 
                     quadruple const *y, idx_int incy);

template
std::complex<quadruple> dot<std::complex<quadruple>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<quadruple> const *x, idx_int incx, 
                     std::complex<double>    const *y, idx_int incy);

template
std::complex<quadruple> dot<std::complex<double>, std::complex<quadruple> >(
                     const idx_int n,  
                     std::complex<double>    const *x, idx_int incx, 
                     std::complex<quadruple> const *y, idx_int incy);

}
