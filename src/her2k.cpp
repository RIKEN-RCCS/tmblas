//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "her2k_tmpl.hpp"

namespace tmblas{

template
void her2k<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<half> &alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half> const *B, idx_int ldb,
    half &beta,
    std::complex<half>       *C, idx_int ldc,
    std::complex<float>      *w);

template
void her2k<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const *B, idx_int ldb,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<double>      *w);

template
void her2k<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const *B, idx_int ldb,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<quadruple>      *w);

#ifdef CBLAS_ROUTINES

template<>
void her2k<std::complex<float>, std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const *B, idx_int ldb,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w)
{
  cblas_cher2k(CblasColMajor, 
               uplo2cblas(uplo), 
               op2cblas(trans), 
               (BLAS_INT)n, (BLAS_INT)k, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)B, (BLAS_INT)ldb, beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}

template<>
void her2k<std::complex<double>, std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const *B, idx_int ldb,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w)
{
  cblas_zher2k(CblasColMajor, 
               uplo2cblas(uplo), 
               op2cblas(trans), 
               (BLAS_INT)n, (BLAS_INT)k, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID const *)B, (BLAS_INT)ldb, beta, (BLAS_VOID *)C, (BLAS_INT)ldc);
}
#else

template
void her2k<std::complex<float>, std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const *B, idx_int ldb,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void her2k<std::complex<double>, std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const *B, idx_int ldb,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w);
#endif  // #ifdef CBLAS_ROUTINES

template
void her2k<std::complex<half>, std::complex<half>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<half> &alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half> const *B, idx_int ldb,
    half &beta,
    std::complex<half>       *C, idx_int ldc,
    std::complex<half>       *w);

template
void her2k<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<quadruple> const *B, idx_int ldb,
    quadruple &beta,
    std::complex<quadruple>       *C, idx_int ldc,
    std::complex<quadruple>       *w);

template
void her2k<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<octuple> &alpha,
    std::complex<octuple> const *A, idx_int lda,
    std::complex<octuple> const *B, idx_int ldb,
    octuple &beta,
    std::complex<octuple>       *C, idx_int ldc,
    std::complex<octuple>       *w);

template
void her2k<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<half>  const *A, idx_int lda,
    std::complex<half>  const *B, idx_int ldb,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void her2k<std::complex<half>, std::complex<float>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<half>  const *A, idx_int lda,
    std::complex<float> const *B, idx_int ldb,
    float &beta,
    std::complex<half>        *C, idx_int ldc,
    std::complex<float>       *w);

template
void her2k<std::complex<float>, std::complex<half>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<half>  const *B, idx_int ldb,
    float &beta,
    std::complex<half>        *C, idx_int ldc,
    std::complex<float>       *w);

template
void her2k<std::complex<half>, std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<half>  const *A, idx_int lda,
    std::complex<float>  const *B, idx_int ldb,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void her2k<std::complex<float>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<half>  const *B, idx_int ldb,
    float &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<float>       *w);

template
void her2k<std::complex<float>, std::complex<float>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<float> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const *B, idx_int ldb,
    float &beta,
    std::complex<half>        *C, idx_int ldc,
    std::complex<float>       *w);

template
void her2k<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float> const *B, idx_int ldb,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w);

template
void her2k<std::complex<float>, std::complex<double>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<double> const *B, idx_int ldb,
    double &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<double>       *w);

template
void her2k<std::complex<double>, std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<float> const *B, idx_int ldb,
    double &beta,
    std::complex<float>       *C, idx_int ldc,
    std::complex<double>       *w);

template
void her2k<std::complex<double>, std::complex<double>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const *B, idx_int ldb,
    double &beta,
    std::complex<float>        *C, idx_int ldc,
    std::complex<double>       *w);

template
void her2k<std::complex<double>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<float>  const *B, idx_int ldb,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w);

template
void her2k<std::complex<float>, std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<double> &alpha,
    std::complex<float>  const *A, idx_int lda,
    std::complex<double> const *B, idx_int ldb,
    double &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<double>       *w);

template
void her2k<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double> const *B, idx_int ldb,
    quadruple &beta,
    std::complex<quadruple>    *C, idx_int ldc,
    std::complex<quadruple>    *w);

template
void her2k<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> &alpha,
    std::complex<double>    const *A, idx_int lda,
    std::complex<quadruple> const *B, idx_int ldb,
    quadruple &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<quadruple>    *w);

template
void her2k<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<double>    const *B, idx_int ldb,
    quadruple &beta,
    std::complex<double>       *C, idx_int ldc,
    std::complex<quadruple>    *w);

template
void her2k<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<quadruple> const *B, idx_int ldb,
    quadruple &beta,
    std::complex<double>          *C, idx_int ldc,
    std::complex<quadruple>       *w);

template
void her2k<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<double>    const *B, idx_int ldb,
    quadruple &beta,
    std::complex<quadruple>       *C, idx_int ldc,
    std::complex<quadruple>       *w);

template
void her2k<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    idx_int n, idx_int k,
    std::complex<quadruple> &alpha,
    std::complex<double>    const *A, idx_int lda,
    std::complex<quadruple> const *B, idx_int ldb,
    quadruple &beta,
    std::complex<quadruple>       *C, idx_int ldc,
    std::complex<quadruple>       *w);

}

