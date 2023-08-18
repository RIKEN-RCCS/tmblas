//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _OMATCOPY_HPP
#define _OMATCOPY_HPP

namespace tmblas {

template< typename Ta, typename Tb, typename Td = blas::scalar_type<Ta, Tb> >
void omatcopy(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    blas::scalar_type<Ta, Tb> alpha,
    Ta const *A, idx_int lda,
    Tb       *B, idx_int ldb);

#ifdef MKL_OMATCOPY
template<>
void omatcopy<float, float>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    float alpha,
    float const *A, idx_int lda,
    float       *B, idx_int ldb);

template<>
void omatcopy<double, double>(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    double alpha,
    double const *A, idx_int lda,
    double       *B, idx_int ldb);

template<>
void omatcopy<std::complex<float>, std::complex<float> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<float> alpha,
    std::complex<float> const *A, idx_int lda,
    std::complex<float>       *B, idx_int ldb);

template<>
void omatcopy<std::complex<double>, std::complex<double> >(
    blas::Op trans,
    idx_int rows,
    idx_int cols,
    std::complex<double> alpha,
    std::complex<double> const *A, idx_int lda,
    std::complex<double>       *B, idx_int ldb);
#endif

}

#endif
