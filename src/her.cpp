//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "her_tmpl.hpp"

namespace tmblas {

template 
void her<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    half const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half>       *A, idx_int lda);

template 
void her<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void her<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double>       *A, idx_int lda);

#ifdef CBLAS_ROUTINES
// her with complex<float> == cher
template<>
void her<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda)
{
  cblas_cher(CblasColMajor, uplo2cblas(uplo), n, alpha, (BLAS_VOID const *)x, incx, (BLAS_VOID *)A, lda);
}
// her with complex<double> = zher
template<>
void her<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double>       *A, idx_int lda)
{
  cblas_zher(CblasColMajor, uplo2cblas(uplo), n, alpha, (BLAS_VOID const *)x, incx, (BLAS_VOID *)A, lda);
}

#else

template 
void her<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void her<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double>       *A, idx_int lda);
#endif

template 
void her<std::complex<half>, std::complex<half> >(
    blas::Uplo uplo,
    idx_int n,
    half const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half>       *A, idx_int lda);

template 
void her<std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo uplo,
    idx_int n,
    quadruple const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<quadruple>       *A, idx_int lda);

template 
void her<std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo uplo,
    idx_int n,
    octuple const &alpha,
    std::complex<octuple> const *x, idx_int incx,
    std::complex<octuple>       *A, idx_int lda);

template 
void her<std::complex<float>, std::complex<half> >(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<half>       *A, idx_int lda);

template 
void her<std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    float const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void her<std::complex<double>, std::complex<float> >(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<float>       *A, idx_int lda);

template 
void her<std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    double const &alpha,
    std::complex<float> const  *x, idx_int incx,
    std::complex<double>       *A, idx_int lda);

template 
void her<std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    idx_int n,
    quadruple const &alpha,
    std::complex<double> const  *x, idx_int incx,
    std::complex<quadruple>     *A, idx_int lda);

template 
void her<std::complex<quadruple>, std::complex<double> >(
    blas::Uplo uplo,
    idx_int n,
    quadruple const &alpha,
    std::complex<quadruple> const  *x, idx_int incx,
    std::complex<double>        *A, idx_int lda);

}

