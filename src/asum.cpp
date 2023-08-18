//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "asum_tmpl.hpp"

namespace tmblas {

template 
half asum<half, float>(
                     idx_int n,  
                     half const *x, idx_int incx);

template 
half asum<std::complex<half>, std::complex<float> >(
                     idx_int n,  
                     std::complex<half> const *x, idx_int incx);

template 
float asum<float, double>(
                     idx_int n,  
                     float const *x, idx_int incx);

template 
float asum<std::complex<float>, std::complex<double> >(
                     idx_int n,  
                     std::complex<float> const *x, idx_int incx);

template 
double asum<double, quadruple>(
                     idx_int n,  
                     double const *x, idx_int incx);

template 
double asum<std::complex<double>, std::complex<quadruple> >(
                     idx_int n,  
                     std::complex<double> const *x, idx_int incx);

#ifdef CBLAS_ROUTINES
template<>
float asum<float>(
		  idx_int n,  
		  float const *x, idx_int incx)
{
  return cblas_sasum(n, x, (BLAS_INT) incx);
}

template<>
double asum<double>(
                     idx_int n,  
                     double const *x, idx_int incx)
{
  return cblas_dasum(n, x, (BLAS_INT) incx);
}


template<>
float asum<std::complex<float> >(
                     idx_int n,  
                     std::complex<float> const *x, idx_int incx)
{
  float result;//std::complex<float> result;
  result = cblas_scasum(n, (BLAS_VOID const *)x, (BLAS_INT) incx); //cblas_casumc_sub(n, x, (BLAS_INT) incx, &result);
  return result;
}

template<>
double asum<std::complex<double>>(
                     idx_int n,  
                     std::complex<double> const *x, idx_int incx)
{
  double result;//std::complex<double> result;
  result = cblas_dzasum(n, (BLAS_VOID const *)x, (BLAS_INT) incx); //cblas_zasumc_sub(n, x, (BLAS_INT) incx, &result);
  return result;
}

#else
template
float asum<float>(idx_int n,  
		  float const *x, idx_int incx); 
  
template
double asum<double>(idx_int n,  
			    double const *x, idx_int incx);

template
float asum<std::complex<float> >(
                     idx_int n,  
                     std::complex<float> const *x, idx_int incx);

template
double asum<std::complex<double> >(
                     idx_int n,  
                     std::complex<double> const *x, idx_int incx);
#endif

template
half asum<half>( idx_int n,  
                 half const *x, idx_int incx);

template
half asum<std::complex<half> >(
                     idx_int n,  
                     std::complex<half> const *x, idx_int incx);

template
quadruple asum<quadruple>(
                     idx_int n,  
                     quadruple const *x, idx_int incx);

template
quadruple asum<std::complex<quadruple> >(
                     idx_int n,  
                     std::complex<quadruple> const *x, idx_int incx);
template
octuple asum<octuple>(
                     idx_int n,  
                     octuple const *x, idx_int incx);
  
template
octuple asum<std::complex<octuple> >(
                     idx_int n,  
                     std::complex<octuple> const *x, idx_int incx);

}
