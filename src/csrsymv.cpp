//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "csrsymv_tmpl.hpp"

namespace tmblas{
//real
//  HHH
template
void csrsymv<half, half, half>(
     blas::Uplo uplo,
     idx_int m,
     half const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     half const *x,
     half const &beta,
     half *y,
     half *w);
//  HHHF
template
void csrsymv<half, half, half, float>(
     blas::Uplo uplo,
     idx_int m,
     half const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     half const *x,
     half const &beta,
     half *y,
     float *w);
//  HHF ?
template
void csrsymv<half, half, float>(
     blas::Uplo uplo,
     idx_int m,
     float const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     half const *x,
     float const &beta,
     float *y,
     float *w);
//  HFH ?
template
void csrsymv<half, float, half>(
     blas::Uplo uplo,
     idx_int m,
     float const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     float const &beta,
     half *y,
     float *w);
// HFF
template
void csrsymv<half, float, float>(
     blas::Uplo uplo,
     idx_int m,
     float const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     float const &beta,
     float *y,
     float *w);

//  FFF
template
void csrsymv<float, float, float>(
     blas::Uplo uplo,
     idx_int m,
     float const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     float const &beta,
     float *y,
     float *w);
//  FFFD
template
void csrsymv<float, float, float, double>(
     blas::Uplo uplo,
     idx_int m,
     float const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     float const &beta,
     float *y,
     double *w);
//  FFD ?
template
void csrsymv<float, float, double>(
     blas::Uplo uplo,
     idx_int m,
     double const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     double const &beta,
     double *y,
     double *w);
//  FDF ?
template
void csrsymv<float, double, float>(
     blas::Uplo uplo,
     idx_int m,
     double const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     double const &beta,
     float *y,
     double *w);
// FDD
template
void csrsymv<float, double, double>(
     blas::Uplo uplo,
     idx_int m,
     double const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     double const &beta,
     double *y,
     double *w);

//  DDD
template
void csrsymv<double, double, double>(
     blas::Uplo uplo,
     idx_int m,
     double const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     double const &beta,
     double *y,
     double *w);
//  DDDQ
template
void csrsymv<double, double, double, quadruple>(
     blas::Uplo uplo,
     idx_int m,
     double const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     double const &beta,
     double *y,
     quadruple *w);
//  DDQ ?
template
void csrsymv<double, double, quadruple>(
     blas::Uplo uplo,
     idx_int m,
     quadruple const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     quadruple const &beta,
     quadruple *y,
     quadruple *w);
//  DQD ?
template
void csrsymv<double, quadruple, double>(
     blas::Uplo uplo,
     idx_int m,
     quadruple const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     quadruple const &beta,
     double *y,
     quadruple *w);
// DQQ
template
void csrsymv<double, quadruple, quadruple>(
     blas::Uplo uplo,
     idx_int m,
     quadruple const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     quadruple const &beta,
     quadruple*y,
     quadruple *w);

//  QQQ
template
void csrsymv<quadruple, quadruple, quadruple>(
     blas::Uplo uplo,
     idx_int m,
     quadruple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     quadruple const &beta,
     quadruple *y,
     quadruple *w);
//  QQQO
template
void csrsymv<quadruple, quadruple, quadruple, octuple>(
     blas::Uplo uplo,
     idx_int m,
     quadruple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     quadruple const &beta,
     quadruple *y,
     octuple *w);
//  QQO ?
template
void csrsymv<quadruple, quadruple, octuple>(
     blas::Uplo uplo,
     idx_int m,
     octuple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     octuple const &beta,
     octuple *y,
     octuple *w);
//  QOQ ?
template
void csrsymv<quadruple, octuple, quadruple>(
     blas::Uplo uplo,
     idx_int m,
     octuple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     octuple const *x,
     octuple const &beta,
     quadruple *y,
     octuple *w);
// QOO
template
void csrsymv<quadruple, octuple, octuple>(
     blas::Uplo uplo,
     idx_int m,
     octuple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     octuple const *x,
     octuple const &beta,
     octuple*y,
     octuple *w);
// OOO
template
void csrsymv<octuple, octuple, octuple>(
     blas::Uplo uplo,
     idx_int m,
     octuple const &alpha,     
     octuple const *av,
     idx_int const *ai,
     idx_int const *aj,
     octuple const *x,
     octuple const &beta,
     octuple *y,
     octuple *w);

//complex
//  HHH
template
void csrsymv<std::complex<half>, std::complex<half>, std::complex<half> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<half> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<half> const *x,
     std::complex<half> const &beta,
     std::complex<half> *y,
     std::complex<half> *w);
//  HHHF
template
void csrsymv<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<half> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<half> const *x,
     std::complex<half> const &beta,
     std::complex<half> *y,
     std::complex<float> *w);
//  HHF ?
template
void csrsymv<std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<float> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<half> const *x,
     std::complex<float> const &beta,
     std::complex<float> *y,
     std::complex<float> *w);
//  HFH ?
template
void csrsymv<std::complex<half>, std::complex<float>, std::complex<half> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<float> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     std::complex<float> const &beta,
     std::complex<half> *y,
     std::complex<float> *w);
// HFF
template
void csrsymv<std::complex<half>, std::complex<float>, std::complex<float> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<float> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     std::complex<float> const &beta,
     std::complex<float> *y,
     std::complex<float> *w);

//  FFF
template
void csrsymv<std::complex<float>, std::complex<float>, std::complex<float> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<float> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     std::complex<float> const &beta,
     std::complex<float> *y,
     std::complex<float> *w);
//  FFFD
template
void csrsymv<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<float> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     std::complex<float> const &beta,
     std::complex<float> *y,
     std::complex<double> *w);
//  FFD ?
template
void csrsymv<std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<double> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     std::complex<double> const &beta,
     std::complex<double> *y,
     std::complex<double> *w);
//  FDF ?
template
void csrsymv<std::complex<float>, std::complex<double>, std::complex<float> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<double> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     std::complex<double> const &beta,
     std::complex<float> *y,
     std::complex<double> *w);
// FDD
template
void csrsymv<std::complex<float>, std::complex<double>, std::complex<double> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<double> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     std::complex<double> const &beta,
     std::complex<double> *y,
     std::complex<double> *w);

//  DDD
template
void csrsymv<std::complex<double>, std::complex<double>, std::complex<double> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<double> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     std::complex<double> const &beta,
     std::complex<double> *y,
     std::complex<double> *w);
//  DDDQ
template
void csrsymv<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<double> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     std::complex<double> const &beta,
     std::complex<double> *y,
     std::complex<quadruple> *w);
//  DDQ ?
template
void csrsymv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<quadruple> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     std::complex<quadruple> *w);
//  DQD ?
template
void csrsymv<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<quadruple> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     std::complex<quadruple> const &beta,
     std::complex<double> *y,
     std::complex<quadruple> *w);
// DQQ
template
void csrsymv<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<quadruple> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     std::complex<quadruple> *w);

//  QQQ
template
void csrsymv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<quadruple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     std::complex<quadruple> *w);
//  QQQO
template
void csrsymv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<quadruple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     std::complex<octuple> *w);
//  QQO ?
template
void csrsymv<std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<octuple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     std::complex<octuple> const &beta,
     std::complex<octuple> *y,
     std::complex<octuple> *w);
//  QOQ ?
template
void csrsymv<std::complex<quadruple>, std::complex<octuple>, std::complex<quadruple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<octuple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<octuple> const *x,
     std::complex<octuple> const &beta,
     std::complex<quadruple> *y,
     std::complex<octuple> *w);
// QOO
template
void csrsymv<std::complex<quadruple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<octuple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<octuple> const *x,
     std::complex<octuple> const &beta,
     std::complex<octuple>*y,
     std::complex<octuple> *w);
// OOO
template
void csrsymv<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Uplo uplo,
     idx_int m,
     std::complex<octuple> const &alpha,     
     std::complex<octuple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<octuple> const *x,
     std::complex<octuple> const &beta,
     std::complex<octuple> *y,
     std::complex<octuple> *w);
  
  
} // namespace tmblas

