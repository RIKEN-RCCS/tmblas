//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "syr2_tmpl.hpp"

namespace tmblas {

template
void syr2<half, half, half, float>(
    blas::Uplo  uplo,
    idx_int n,
    half const &alpha,
    half const *x, idx_int incx,
    half const *y, idx_int incy,
    half *A, idx_int lda);

template
void syr2<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void syr2<float, float, float, double>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    float const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<double, double, double, quadruple>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double const *y, idx_int incy,
    double *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

#ifdef CBLAS_ROUTINES
template<>
void syr2<float, float, float>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    float const *y, idx_int incy,
    float *A, idx_int lda)
{
  cblas_ssyr2(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, alpha, x, (BLAS_INT)incx, y, (BLAS_INT)incy, A, (BLAS_INT)lda);
}

template<>
void syr2<double, double, double>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double const *y, idx_int incy,
    double *A, idx_int lda)
{
  cblas_dsyr2(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, alpha, x, (BLAS_INT)incx, y, (BLAS_INT)incy, A, (BLAS_INT)lda);
}

#else
template
void syr2<float, float, float>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    float const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<double, double, double>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double const *y, idx_int incy,
    double *A, idx_int lda);
#endif

  
template
void syr2<std::complex<float>, std::complex<float>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<double>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void syr2<half, half, half>(
    blas::Uplo  uplo,
    idx_int n,
    half const &alpha,
    half const *x, idx_int incx,
    half const *y, idx_int incy,
    half *A, idx_int lda);

template
void syr2<std::complex<half>, std::complex<half>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void syr2<quadruple, quadruple, quadruple>(
    blas::Uplo  uplo,
    idx_int n,
    quadruple const &alpha,
    quadruple const *x, idx_int incx,
    quadruple const *y, idx_int incy,
    quadruple *A, idx_int lda);

template
void syr2<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

template
void syr2<octuple, octuple, octuple>(
    blas::Uplo  uplo,
    idx_int n,
    octuple const &alpha,
    octuple const *x, idx_int incx,
    octuple const *y, idx_int incy,
    octuple *A, idx_int lda);

template
void syr2<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<octuple> const &alpha,
    std::complex<octuple> const *x, idx_int incx,
    std::complex<octuple> const *y, idx_int incy,
    std::complex<octuple> *A, idx_int lda);

template
void syr2<half, half, float>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    half const *x, idx_int incx,
    half const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<half, float, half>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    half const *x, idx_int incx,
    float const *y, idx_int incy,
    half *A, idx_int lda);

template
void syr2<float, half, half>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    half  const *y, idx_int incy,
    half *A, idx_int lda);

template
void syr2<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<std::complex<half>, std::complex<float>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void syr2<std::complex<float>, std::complex<half>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<half>  const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void syr2<half, float, float>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    half const *x, idx_int incx,
    float const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<float, half, float>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    half const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<float, float, half>(
    blas::Uplo  uplo,
    idx_int n,
    float const &alpha,
    float const *x, idx_int incx,
    float const *y, idx_int incy,
    half *A, idx_int lda);

template
void syr2<std::complex<half>, std::complex<float>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<std::complex<float>, std::complex<half>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<std::complex<float>, std::complex<float>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void syr2<float, float, double>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    float const *x, idx_int incx,
    float const *y, idx_int incy,
    double *A, idx_int lda);

template
void syr2<float, double, float>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    float const *x, idx_int incx,
    double const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<double, float, float>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    float const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void syr2<std::complex<float>, std::complex<double>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<float>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<double, double, float>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    double const *y, idx_int incy,
    float *A, idx_int lda);

template
void syr2<double, float, double>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    double const *x, idx_int incx,
    float const *y, idx_int incy,
    double *A, idx_int lda);

template
void syr2<float, double, double>(
    blas::Uplo  uplo,
    idx_int n,
    double const &alpha,
    float const *x, idx_int incx,
    double const *y, idx_int incy,
    double *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<double>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<float>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void syr2<std::complex<float>, std::complex<double>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void syr2<double, double, quadruple>(
    blas::Uplo  uplo,
    idx_int n,
    quadruple const &alpha,
    double const *x, idx_int incx,
    double const *y, idx_int incy,
    quadruple *A, idx_int lda);

template
void syr2<double, quadruple, double>(
    blas::Uplo  uplo,
    idx_int n,
    quadruple const &alpha,
    double const *x, idx_int incx,
    quadruple const *y, idx_int incy,
    double *A, idx_int lda);

template
void syr2<quadruple, double, double>(
    blas::Uplo  uplo,
    idx_int n,
    quadruple const &alpha,
    quadruple const *x, idx_int incx,
    double const *y, idx_int incy,
    double *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void syr2<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void syr2<quadruple, quadruple, double>(
    blas::Uplo  uplo,
    idx_int n,
    quadruple const &alpha,
    quadruple const *x, idx_int incx,
    quadruple const *y, idx_int incy,
    double *A, idx_int lda);

template
void syr2<quadruple, double, quadruple>(
    blas::Uplo  uplo,
    idx_int n,
    quadruple const &alpha,
    quadruple const *x, idx_int incx,
    double const *y, idx_int incy,
    quadruple *A, idx_int lda);

template
void syr2<double, quadruple, quadruple>(
    blas::Uplo  uplo,
    idx_int n,
    quadruple const &alpha,
    double const *x, idx_int incx,
    quadruple const *y, idx_int incy,
    quadruple *A, idx_int lda);

template
void syr2<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void syr2<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

template
void syr2<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

}

