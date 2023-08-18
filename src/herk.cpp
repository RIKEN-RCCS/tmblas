//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "herk_tmpl.hpp"

namespace tmblas {

template
void herk<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    half &alpha,
    std::complex<half> const *A, idx_int lda,
    half &beta,
    std::complex<half>       *C, idx_int ldc,
    std::complex<float>      *w);

template
void herk<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float &alpha,
    std::complex<float> const *A, idx_int lda,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<double>      *w);

#ifdef CBLAS_ROUTINES

template<>
void herk<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float &alpha,
    std::complex<float> const *A, idx_int lda,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w)
{
  cblas_cherk(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), (BLAS_INT)n, (BLAS_INT)k, alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}

  template<>
void herk<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double &alpha,
    std::complex<double> const *A, idx_int lda,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w)
{
  cblas_zherk(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), (BLAS_INT)n, (BLAS_INT)k, alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}
#else

template
void herk<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float &alpha,
    std::complex<float> const *A, idx_int lda,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void herk<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double &alpha,
    std::complex<double> const *A, idx_int lda,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w);

#endif

template
void herk<std::complex<half>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    half &alpha,
    std::complex<half> const *A, idx_int lda,
    half &beta,
    std::complex<half>       *C, idx_int ldc,
    std::complex<half>       *w);

template
void herk<std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float &alpha,
    std::complex<half> const  *A, idx_int lda,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void herk<std::complex<float>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float &alpha,
    std::complex<float> const *A, idx_int lda,
    float &beta,
    std::complex<half>        *C, idx_int ldc,
    std::complex<float>       *w);

template
void herk<std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    quadruple &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    quadruple &beta,
    std::complex<quadruple>       *C, idx_int ldc,
    std::complex<quadruple>       *w);

template
void herk<std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    octuple &alpha,
    std::complex<octuple> const *A, idx_int lda,
    octuple &beta,
    std::complex<octuple>       *C, idx_int ldc,
    std::complex<octuple>       *w);

template
void herk<std::complex<double>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double &alpha,
    std::complex<double> const  *A, idx_int lda,
    double &beta,
    std::complex<float>        *C, idx_int ldc,
    std::complex<double>       *w);

template
void herk<std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double &alpha,
    std::complex<float> const  *A, idx_int lda,
    double &beta,
    std::complex<double>        *C, idx_int ldc,
    std::complex<double>       *w);

template
void herk<std::complex<quadruple>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    quadruple &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    quadruple &beta,
    std::complex<double>          *C, idx_int ldc,
    std::complex<quadruple>       *w);

template
void herk<std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    quadruple &alpha,
    std::complex<double> const    *A, idx_int lda,
    quadruple &beta,
    std::complex<quadruple>       *C, idx_int ldc,
    std::complex<quadruple>       *w);
}

