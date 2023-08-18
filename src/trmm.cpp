//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "trmm_tmpl.hpp"

namespace tmblas{

template
void trmm<half, half, float>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    half const &alpha,
    half const *A, idx_int lda,
    half       *B, idx_int ldb,
    float      *w);

template
void trmm<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half>       *B, idx_int ldb,
    std::complex<float>      *w);

template
void trmm<float, float, double>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    float const &alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb,
    double      *w);

template
void trmm<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb,
    std::complex<double>      *w);

template
void trmm<double, double, quadruple>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    double const &alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb,
    quadruple    *w);

template
void trmm<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb,
    std::complex<quadruple>      *w);

#ifdef CBLAS_ROUTINES
template<>
void trmm<float, float>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    float const &alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb,
    float       *w)
{
  cblas_strmm(CblasColMajor, 
              side2cblas(side), 
              uplo2cblas(uplo), 
              op2cblas(trans), 
              diag2cblas(diag), 
              (BLAS_INT)m, (BLAS_INT)n, alpha, A, (BLAS_INT)lda, B, (BLAS_INT)ldb);
}

template<>
void trmm<double, double>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    double const &alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb,
    double       *w)
{
  cblas_dtrmm(CblasColMajor, 
              side2cblas(side), 
              uplo2cblas(uplo), 
              op2cblas(trans), 
              diag2cblas(diag), 
              (BLAS_INT)m, (BLAS_INT)n, alpha, A, (BLAS_INT)lda, B, (BLAS_INT)ldb);
}

template<>
void trmm<std::complex<float>, std::complex<float> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb,
    std::complex<float>       *w)
{
  cblas_ctrmm(CblasColMajor, 
              side2cblas(side), 
              uplo2cblas(uplo), 
              op2cblas(trans), 
              diag2cblas(diag), 
              (BLAS_INT)m, (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID *)B, (BLAS_INT)ldb);
}

template<>
void trmm<std::complex<double>, std::complex<double> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb,
    std::complex<double>       *w)
{
  cblas_ztrmm(CblasColMajor, 
              side2cblas(side), 
              uplo2cblas(uplo), 
              op2cblas(trans), 
              diag2cblas(diag), 
              (BLAS_INT)m, (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID *)B, (BLAS_INT)ldb);
}
#else
template
void trmm<float, float>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    float const &alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb,
    float       *w);

template
void trmm<double, double>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    double const &alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb,
    double       *w);

template
void trmm<std::complex<float>, std::complex<float> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb,
    std::complex<float>       *w);

template
void trmm<std::complex<double>, std::complex<double> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb,
    std::complex<double>       *w);
#endif
  
template
void trmm<half, half>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    half const &alpha,
    half const *A, idx_int lda,
    half       *B, idx_int ldb,
    half       *w);

template
void trmm<std::complex<half>, std::complex<half> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half>       *B, idx_int ldb,
    std::complex<half>       *w);

template
void trmm<quadruple, quadruple>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    quadruple const &alpha,
    quadruple const *A, idx_int lda,
    quadruple       *B, idx_int ldb,
    quadruple       *w);

template
void trmm<std::complex<quadruple>, std::complex<quadruple> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<quadruple>       *B, idx_int ldb,
    std::complex<quadruple>       *w);

template
void trmm<octuple, octuple>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    octuple const &alpha,
    octuple const *A, idx_int lda,
    octuple       *B, idx_int ldb,
    octuple       *w);

template
void trmm<std::complex<octuple>, std::complex<octuple> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<octuple> const &alpha,
    std::complex<octuple> const *A, idx_int lda,
    std::complex<octuple>       *B, idx_int ldb,
    std::complex<octuple>       *w);

template
void trmm<float, half>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    float const &alpha,
    float const *A, idx_int lda,
    half        *B, idx_int ldb,
    float       *w);

template
void trmm<half, float>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    float const &alpha,
    half const  *A, idx_int lda,
    float       *B, idx_int ldb,
    float       *w);

template
void trmm<std::complex<float>, std::complex<half> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<half>        *B, idx_int ldb,
    std::complex<float>       *w);

template
void trmm<std::complex<half>, std::complex<float> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const  *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb,
    std::complex<float>       *w);

template
void trmm<double, float>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    double const &alpha,
    double const *A, idx_int lda,
    float        *B, idx_int ldb,
    double       *w);

template
void trmm<float, double>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    double const &alpha,
    float const  *A, idx_int lda,
    double       *B, idx_int ldb,
    double       *w);

template
void trmm<std::complex<double>, std::complex<float> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<float>        *B, idx_int ldb,
    std::complex<double>       *w);

template
void trmm<std::complex<float>, std::complex<double> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const  *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb,
    std::complex<double>       *w);

template
void trmm<double, quadruple>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    quadruple const &alpha,
    double const *A, idx_int lda,
    quadruple    *B, idx_int ldb,
    quadruple    *w);

template
void trmm<quadruple, double>(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    quadruple const &alpha,
    quadruple const *A, idx_int lda,
    double          *B, idx_int ldb,
    quadruple       *w);

template
void trmm<std::complex<double>, std::complex<quadruple> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<quadruple>    *B, idx_int ldb,
    std::complex<quadruple>    *w);

template
void trmm<std::complex<quadruple>, std::complex<double> >(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int m,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<double>          *B, idx_int ldb,
    std::complex<quadruple>       *w);

}

