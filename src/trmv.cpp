//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "trmv_tmpl.hpp"

namespace tmblas {

template
void trmv<half, half, float>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    half const *A, idx_int lda,
    half       *x, idx_int incx,
    float *w);

template
void trmv<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<half> const *A, idx_int lda,
    std::complex<half>       *x, idx_int incx,
    std::complex<float> * w);

template
void trmv<float, float, double>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    float const *A, idx_int lda,
    float       *x, idx_int incx,
    double *w);

template
void trmv<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *x, idx_int incx,
    std::complex<double>       *w);

template
void trmv<double, double, quadruple>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    double const *A, idx_int lda,
    double       *x, idx_int incx,
    quadruple    *w);

template
void trmv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *x, idx_int incx,
    std::complex<quadruple>    *w);

#ifdef CBLAS_ROUTINES
template<>
void trmv<float, float>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    float const *A, idx_int lda,
    float       *x, idx_int incx,
    float       *w )
{
  cblas_strmv(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), diag2cblas(diag), (BLAS_INT)n, A, (BLAS_INT)lda, x, (BLAS_INT)incx);
}

template<>
void trmv<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *x, idx_int incx,
    std::complex<float>       *w)
{
  cblas_ctrmv(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), diag2cblas(diag), (BLAS_INT)n, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID *)x, (BLAS_INT)incx);
}

template<>
void trmv<double, double>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    double const *A, idx_int lda,
    double       *x, idx_int incx,
    double       *w)
{
  cblas_dtrmv(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), diag2cblas(diag), (BLAS_INT)n, A, (BLAS_INT)lda, x, (BLAS_INT)incx);
}

template<>
void trmv<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *x, idx_int incx,
    std::complex<double>       *w)
{
  cblas_ztrmv(CblasColMajor, uplo2cblas(uplo), op2cblas(trans), diag2cblas(diag), (BLAS_INT)n, (BLAS_VOID const *)A, (BLAS_INT)lda, (BLAS_VOID *)x, (BLAS_INT)incx);
}
#else

template
void trmv<float, float>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    float const *A, idx_int lda,
    float       *x, idx_int incx,
    float       *w);

template
void trmv<std::complex<float>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *x, idx_int incx,
    std::complex<float>       *w);

template
void trmv<double, double>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    double const *A, idx_int lda,
    double       *x, idx_int incx,
    double       *w);

template
void trmv<std::complex<double>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *x, idx_int incx,
    std::complex<double>       *w);
#endif

template
void trmv<half, half>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    half const *A, idx_int lda,
    half       *x, idx_int incx,
    half       *w);

template
void trmv<std::complex<half>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<half> const *A, idx_int lda,
    std::complex<half>       *x, idx_int incx,
    std::complex<half>       *w);

template
void trmv<quadruple, quadruple>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    quadruple const *A, idx_int lda,
    quadruple       *x, idx_int incx,
    quadruple       *w);

template
void trmv<std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<quadruple>       *x, idx_int incx,
    std::complex<quadruple>       *w);

template
void trmv<octuple, octuple>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    octuple const *A, idx_int lda,
    octuple       *x, idx_int incx,
    octuple       *w);

template
void trmv<std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<octuple> const *A, idx_int lda,
    std::complex<octuple>       *x, idx_int incx,
    std::complex<octuple>       *w);

template
void trmv<float, half>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    float const *A, idx_int lda,
    half        *x, idx_int incx,
    float *w);

template
void trmv<half, float>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    half const *A, idx_int lda,
    float      *x, idx_int incx,
    float      *w);

template
void trmv<std::complex<float>, std::complex<half> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<float> const *A, idx_int lda,
    std::complex<half>        *x, idx_int incx,
    std::complex<float>        *w);

template
void trmv<std::complex<half>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<half> const *A, idx_int lda,
    std::complex<float>      *x, idx_int incx,
    std::complex<float>      *w);

template
void trmv<double, float>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    double const *A, idx_int lda,
    float        *x, idx_int incx,
    double        *w);

template
void trmv<float, double>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    float const *A, idx_int lda,
    double      *x, idx_int incx,
    double      *w);

template
void trmv<std::complex<double>, std::complex<float> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<double> const *A, idx_int lda,
    std::complex<float>        *x, idx_int incx,
    std::complex<double>        *w);

template
void trmv<std::complex<float>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<float> const *A, idx_int lda,
    std::complex<double>      *x, idx_int incx,
    std::complex<double>      *w);

template
void trmv<quadruple, double>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    quadruple const *A, idx_int lda,
    double          *x, idx_int incx,
    quadruple          *w);

template
void trmv<double, quadruple>(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    double const *A, idx_int lda,
    quadruple    *x, idx_int incx,
    quadruple    *w);

template
void trmv<std::complex<quadruple>, std::complex<double> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<double>          *x, idx_int incx,
    std::complex<quadruple>          *w);

template
void trmv<std::complex<double>, std::complex<quadruple> >(
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    idx_int n,
    std::complex<double> const *A, idx_int lda,
    std::complex<quadruple>    *x, idx_int incx,
    std::complex<quadruple>    *w);

}

