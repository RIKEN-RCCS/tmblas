//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "syrk_tmpl.hpp"

namespace tmblas {

template
void syrk<half, half, float>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    half const &alpha,
    half const *A, idx_int lda,
    half const &beta,
    half       *C, idx_int ldc,
    float      *w);

template
void syrk<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<half> const &alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half> const &beta,
    std::complex<half>       *C, idx_int ldc,
    std::complex<float>      *w);

template
void syrk<float, float, double>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float const &alpha,
    float const *A, idx_int lda,
    float const &beta,
    float       *C, idx_int ldc,
    double      *w);

template
void syrk<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<double>      *w);

#ifdef CBLAS_ROUTINES
template<>
void syrk<float, float>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float const &alpha,
    float const *A, idx_int lda,
    float const &beta,
    float       *C, idx_int ldc,
    float       *w)
{
  cblas_ssyrk(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), (BLAS_INT)n, (BLAS_INT)k, alpha, A, (BLAS_INT)lda, beta, C, (BLAS_INT)ldc);
}

template<>
void syrk<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w)
{
  cblas_csyrk(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), (BLAS_INT)n, (BLAS_INT)k, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)&beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}

template<>
void syrk<double, double>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double const &alpha,
    double const *A, idx_int lda,
    double const &beta,
    double       *C, idx_int ldc,
    double       *w)
{
  cblas_dsyrk(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), (BLAS_INT)n, (BLAS_INT)k, alpha, A, (BLAS_INT)lda, beta, C, (BLAS_INT)ldc);
}

template<>
void syrk<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w)
{
  cblas_zsyrk(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), (BLAS_INT)n, (BLAS_INT)k, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)&beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}
#else
template
void syrk<float, float>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float const &alpha,
    float const *A, idx_int lda,
    float const &beta,
    float       *C, idx_int ldc,
    float       *w);

template
void syrk<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void syrk<double, double>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double const &alpha,
    double const *A, idx_int lda,
    double const &beta,
    double       *C, idx_int ldc,
    double       *w);

template
void syrk<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w);

#endif

template
void syrk<half, half>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    half const &alpha,
    half const *A, idx_int lda,
    half const &beta,
    half       *C, idx_int ldc,
    half       *w);

template
void syrk<std::complex<half>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<half> const &alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half> const &beta,
    std::complex<half>       *C, idx_int ldc,
    std::complex<half>       *w);

template
void syrk<half, float>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float const &alpha,
    half const  *A, idx_int lda,
    float const &beta,
    float       *C, idx_int ldc,
    float       *w);

template
void syrk<std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> const &alpha,
    std::complex<half> const  *A, idx_int lda,
    std::complex<float> const &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void syrk<float, half>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    float const &alpha,
    float const *A, idx_int lda,
    float const &beta,
    half        *C, idx_int ldc,
    float       *w);


template
void syrk<std::complex<float>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const &beta,
    std::complex<half>        *C, idx_int ldc,
    std::complex<float>       *w);

template
void syrk<quadruple, quadruple>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    quadruple const &alpha,
    quadruple const *A, idx_int lda,
    quadruple const &beta,
    quadruple       *C, idx_int ldc,
    quadruple       *w);

template
void syrk<std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<quadruple> const &beta,
    std::complex<quadruple>       *C, idx_int ldc,
    std::complex<quadruple>       *w);

template
void syrk<octuple, octuple>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    octuple const &alpha,
    octuple const *A, idx_int lda,
    octuple const &beta,
    octuple       *C, idx_int ldc,
    octuple       *w);

template
void syrk<std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<octuple> const &alpha,
    std::complex<octuple> const *A, idx_int lda,
    std::complex<octuple> const &beta,
    std::complex<octuple>       *C, idx_int ldc,
    std::complex<octuple>       *w);

template
void syrk<double, float>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double const &alpha,
    double const  *A, idx_int lda,
    double const &beta,
    float        *C, idx_int ldc,
    double       *w);

template
void syrk<float, double>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    double const &alpha,
    float const  *A, idx_int lda,
    double const &beta,
    double        *C, idx_int ldc,
    double       *w);

template
void syrk<std::complex<double>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> const &alpha,
    std::complex<double> const  *A, idx_int lda,
    std::complex<double> const &beta,
    std::complex<float>        *C, idx_int ldc,
    std::complex<double>       *w);

template
void syrk<std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> const &alpha,
    std::complex<float> const  *A, idx_int lda,
    std::complex<double> const &beta,
    std::complex<double>        *C, idx_int ldc,
    std::complex<double>       *w);

template
void syrk<quadruple, double>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    quadruple const &alpha,
    quadruple const *A, idx_int lda,
    quadruple const &beta,
    double          *C, idx_int ldc,
    quadruple       *w);

template
void syrk<double, quadruple>(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    quadruple const &alpha,
    double const    *A, idx_int lda,
    quadruple const &beta,
    quadruple       *C, idx_int ldc,
    quadruple       *w);

template
void syrk<std::complex<quadruple>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<quadruple> const &beta,
    std::complex<double>          *C, idx_int ldc,
    std::complex<quadruple>       *w);

template
void syrk<std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> const &alpha,
    std::complex<double> const    *A, idx_int lda,
    std::complex<quadruple> const &beta,
    std::complex<quadruple>       *C, idx_int ldc,
    std::complex<quadruple>       *w);
}

