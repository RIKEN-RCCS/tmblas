//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "her2_tmpl.hpp"

namespace tmblas {

template
void her2<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);


template
void her2<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

#ifdef CBLAS_ROUTINES

template<>
void her2<std::complex<float>, std::complex<float>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda)
{
  cblas_cher2(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)x, (BLAS_INT)incx, (BLAS_VOID const *)y, (BLAS_INT)incy, (BLAS_VOID *)A, (BLAS_INT)lda);
}

template<>
void her2<std::complex<double>, std::complex<double>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda)
{
  cblas_zher2(CblasColMajor, uplo2cblas(uplo), (BLAS_INT)n, (BLAS_VOID const *)&alpha, (BLAS_VOID const *)x, (BLAS_INT)incx, (BLAS_VOID const *)y, (BLAS_INT)incy, (BLAS_VOID *)A, (BLAS_INT)lda);
}
#else

template
void her2<std::complex<float>, std::complex<float>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<double>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);
#endif

template
void her2<std::complex<half>, std::complex<half>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<half> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void her2<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

template
void her2<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<octuple> const &alpha,
    std::complex<octuple> const *x, idx_int incx,
    std::complex<octuple> const *y, idx_int incy,
    std::complex<octuple> *A, idx_int lda);

template
void her2<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<half>, std::complex<float>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void her2<std::complex<float>, std::complex<half>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<half>  const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void her2<std::complex<half>, std::complex<float>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<half> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<float>, std::complex<half>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<half> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<float>, std::complex<float>, std::complex<half> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<float> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<half> *A, idx_int lda);

template
void her2<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void her2<std::complex<float>, std::complex<double>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<float>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<double>, std::complex<float> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<float> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<float>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<float> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void her2<std::complex<float>, std::complex<double>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<double> const &alpha,
    std::complex<float> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void her2<std::complex<quadruple>, std::complex<double>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

  template
void her2<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<double> *A, idx_int lda);

template
void her2<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<quadruple> const *x, idx_int incx,
    std::complex<double> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

template
void her2<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
    blas::Uplo  uplo,
    idx_int n,
    std::complex<quadruple> const &alpha,
    std::complex<double> const *x, idx_int incx,
    std::complex<quadruple> const *y, idx_int incy,
    std::complex<quadruple> *A, idx_int lda);

}

