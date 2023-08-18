//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "syr_tmpl.hpp"

namespace tmblas {

template 
void syr<half, half, float>(
    blas::Uplo uplo,
    idx_int n,
    half const &alpha,
    half const *x, idx_int incx,
    half       *A, idx_int lda);

template 
void syr<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half>       *A, idx_int lda);

template 
void syr<float, float, double>(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    float       *A, idx_int lda);

template 
void syr<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void syr<double, double, quadruple>(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double       *A, idx_int lda);

template 
void syr<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double>       *A, idx_int lda);

#ifdef CBLAS_ROUTINES
template<>
void syr<float, float>(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    float       *A, idx_int lda)
{
  cblas_ssyr(CblasColMajor, uplo2cblas(uplo), n, alpha, x, incx, A, lda);
}

template<>
void syr<double, double>(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double       *A, idx_int lda)
{
  cblas_dsyr(CblasColMajor, uplo2cblas(uplo), n, alpha, x, incx, A, lda);
}
// syr with complex<float> == cher
  /*
template<>
void syr<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda)
{
  cblas_cher(CblasColMajor, uplo2cblas(uplo), n, alpha.real(), x, incx, A, lda);
}
  */
// syr with complex<double> = zher
  /*
template<>
void syr<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double>       *A, idx_int lda)
{
  cblas_zher(CblasColMajor, uplo2cblas(uplo), n, alpha.real(), x, incx, A, lda);
}
*/
#else
template 
void syr<float, float>(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    float       *A, idx_int lda);

template 
void syr<double, double>(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double       *A, idx_int lda);
#endif
template 
void syr<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void syr<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double>       *A, idx_int lda);

template 
void syr<half, half>(
    blas::Uplo uplo,
    idx_int n,
    half const &alpha,
    half const *x, idx_int incx,
    half       *A, idx_int lda);

template 
void syr<std::complex<half>, std::complex<half> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half>       *A, idx_int lda);

template 
void syr<quadruple, quadruple>(
    blas::Uplo uplo,
    idx_int n,
    quadruple const &alpha,
    quadruple const *x, idx_int incx,
    quadruple       *A, idx_int lda);

template 
void syr<std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<quadruple>       *A, idx_int lda);

template 
void syr<octuple, octuple>(
    blas::Uplo uplo,
    idx_int n,
    octuple const &alpha,
    octuple const *x, idx_int incx,
    octuple       *A, idx_int lda);

template 
void syr<std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<octuple> const &alpha,
    std::complex<octuple> const *x, idx_int incx,
    std::complex<octuple>       *A, idx_int lda);

template 
void syr<float, half>(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    half       *A, idx_int lda);

template 
void syr<half, float>(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    half const *x, idx_int incx,
    float       *A, idx_int lda);

template 
void syr<std::complex<float>, std::complex<half> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<half>       *A, idx_int lda);

template 
void syr<std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void syr<double, float>(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    float       *A, idx_int lda);

template 
void syr<float, double>(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    float const  *x, idx_int incx,
    double       *A, idx_int lda);

template 
void syr<std::complex<double>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void syr<std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const  *x, idx_int incx,
    std::complex<double>       *A, idx_int lda);

template 
void syr<double, quadruple>(
    blas::Uplo uplo,
    idx_int n,
    quadruple const &alpha,
    double const  *x, idx_int incx,
    quadruple     *A, idx_int lda);

template 
void syr<quadruple, double>(
    blas::Uplo uplo,
    idx_int n,
    quadruple const &alpha,
    quadruple const  *x, idx_int incx,
    double        *A, idx_int lda);

template 
void syr<std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const  *x, idx_int incx,
    std::complex<quadruple>     *A, idx_int lda);

template 
void syr<std::complex<quadruple>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const  *x, idx_int incx,
    std::complex<double>        *A, idx_int lda);

}

