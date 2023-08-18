//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "omatcopy_tmpl.hpp"
#ifdef MKL_OMATCOPY
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include "mkl_trans.h"
#endif

namespace tmblas {

template
void omatcopy<half, half, float>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    half alpha,
    half const *A, idx_int lda,
    half       *B, idx_int ldb);

template
void omatcopy<std::complex<half>, std::complex<half>, std::complex<float> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<half> alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half>       *B, idx_int ldb);

template
void omatcopy<float, float, double>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    float alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb);

template
void omatcopy<std::complex<float>, std::complex<float>, std::complex<double> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<float> alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb);

template
void omatcopy<double, double, quadruple>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    double alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb);

template
void omatcopy<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<double> alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb);

#ifdef MKL_OMATCOPY
template<>
void omatcopy<float, float>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    float alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb)
{
  mkl_somatcopy('C', op2char(trans), rows, cols, alpha, A, lda, B, ldb);
}

template<>
void omatcopy<double, double>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    double alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb)
{
  mkl_domatcopy('C', op2char(trans), rows, cols, alpha, A, lda, B, ldb);
}

template<>
void omatcopy<std::complex<float>, std::complex<float> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<float> alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb)
{
  mkl_comatcopy('C', op2char(trans), rows, cols, alpha, A, lda, B, ldb);
}

template<>
void omatcopy<std::complex<double>, std::complex<double> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<double> alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb)
{
  mkl_zomatcopy('C', op2char(trans), rows, cols, alpha, A, lda, B, ldb);
}
#else
template
void omatcopy<float, float>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    float alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb);

template
void omatcopy<std::complex<float>, std::complex<float> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<float> alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb);

template
void omatcopy<double, double>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    double alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb);

template
void omatcopy<std::complex<double>, std::complex<double> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<double> alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb);
#endif

template
void omatcopy<half, half>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    half alpha,
    half const *A, idx_int lda,
    half       *B, idx_int ldb);

template
void omatcopy<std::complex<half>, std::complex<half> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<half> alpha,
    std::complex<half> const *A, idx_int lda,
    std::complex<half>       *B, idx_int ldb);

template
void omatcopy<quadruple, quadruple>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    quadruple alpha,
    quadruple const *A, idx_int lda,
    quadruple       *B, idx_int ldb);

template
void omatcopy<std::complex<quadruple>, std::complex<quadruple> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<quadruple> alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<quadruple>       *B, idx_int ldb);

template
void omatcopy<octuple, octuple>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    octuple alpha,
    octuple const *A, idx_int lda,
    octuple       *B, idx_int ldb);

template
void omatcopy<std::complex<octuple>, std::complex<octuple> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<octuple> alpha,
    std::complex<octuple> const *A, idx_int lda,
    std::complex<octuple>       *B, idx_int ldb);

template
void omatcopy<half, float>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    float alpha,
    half const  *A, idx_int lda,
    float       *B, idx_int ldb);

template
void omatcopy<float, half>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    float alpha,
    float const *A, idx_int lda,
    half        *B, idx_int ldb);

template
void omatcopy<float, double>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    double alpha,
    float const  *A, idx_int lda,
    double       *B, idx_int ldb);

template
void omatcopy<double, float>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    double alpha,
    double const *A, idx_int lda,
    float        *B, idx_int ldb);

template
void omatcopy<std::complex<float>, std::complex<double> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<double> alpha,
    std::complex<float> const  *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb);

template
void omatcopy<std::complex<double>, std::complex<float> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<double> alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<float>        *B, idx_int ldb);

template
void omatcopy<double, quadruple>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    quadruple alpha,
    double const  *A, idx_int lda,
    quadruple     *B, idx_int ldb);

template
void omatcopy<quadruple, double>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    quadruple alpha,
    quadruple const *A, idx_int lda,
    double          *B, idx_int ldb);

template
void omatcopy<std::complex<double>, std::complex<quadruple> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<quadruple> alpha,
    std::complex<double> const  *A, idx_int lda,
    std::complex<quadruple>     *B, idx_int ldb);

template
void omatcopy<std::complex<quadruple>, std::complex<double> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<quadruple> alpha,
    std::complex<quadruple> const *A, idx_int lda,
    std::complex<double>          *B, idx_int ldb);

}

