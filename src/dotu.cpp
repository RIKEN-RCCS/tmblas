//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "dotu_tmpl.hpp"

namespace tmblas {

template 
std::complex<half> dotu<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<half> const *x, idx_int incx, 
                     std::complex<half> const *y, idx_int incy);

template 
std::complex<float> dotu<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy);

template 
std::complex<double> dotu<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy);

#ifdef CBLAS_ROUTINES

template<>
std::complex<float> dotu<std::complex<float>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy) 
{
  std::complex<float> result;
  cblas_cdotu_sub(n, (BLAS_VOID const *)x, (BLAS_INT) incx, (BLAS_VOID const *)y, (BLAS_INT) incy, (BLAS_VOID *)&result);
  return result;
}

template<>
std::complex<double> dotu<std::complex<double>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy)
{
  std::complex<double> result;
  cblas_zdotu_sub(n, (BLAS_VOID const *)x, (BLAS_INT) incx, (BLAS_VOID const *)y, (BLAS_INT) incy, (BLAS_VOID *)&result);
  return result;
}

#else

  template
std::complex<float> dotu<std::complex<float>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy);

template
std::complex<double> dotu<std::complex<double>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy);
#endif

template
std::complex<half> dotu<std::complex<half>, std::complex<half> >(
                     const idx_int n,  
                     std::complex<half> const *x, idx_int incx, 
                     std::complex<half> const *y, idx_int incy);

template
std::complex<quadruple> dotu<std::complex<quadruple>, std::complex<quadruple> >(
                     const idx_int n,  
                     std::complex<quadruple> const *x, idx_int incx, 
                     std::complex<quadruple> const *y, idx_int incy);
template
std::complex<octuple> dotu<std::complex<octuple>, std::complex<octuple> >(
                     const idx_int n,  
                     std::complex<octuple> const *x, idx_int incx, 
                     std::complex<octuple> const *y, idx_int incy);

template
std::complex<float> dotu<std::complex<float>, std::complex<half> >(
                     const idx_int n,  
                     std::complex<float> const *x, idx_int incx, 
                     std::complex<half>  const *y, idx_int incy);
template
std::complex<float> dotu<std::complex<half>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<half>  const *x, idx_int incx, 
                     std::complex<float> const *y, idx_int incy);

template
std::complex<double> dotu<std::complex<double>, std::complex<float> >(
                     const idx_int n,  
                     std::complex<double> const *x, idx_int incx, 
                     std::complex<float>  const *y, idx_int incy);

template
std::complex<double> dotu<std::complex<float>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<float>  const *x, idx_int incx, 
                     std::complex<double> const *y, idx_int incy);

template
std::complex<quadruple> dotu<std::complex<quadruple>, std::complex<double> >(
                     const idx_int n,  
                     std::complex<quadruple> const *x, idx_int incx, 
                     std::complex<double>    const *y, idx_int incy);

template
std::complex<quadruple> dotu<std::complex<double>, std::complex<quadruple> >(
                     const idx_int n,  
                     std::complex<double>    const *x, idx_int incx, 
                     std::complex<quadruple> const *y, idx_int incy);

}
