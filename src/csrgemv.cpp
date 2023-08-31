//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "csrgemv_tmpl.hpp"

namespace tmblas{
//real
//  HHH
template
void csrgemv<half, half, half>(
     blas::Op trans,
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
void csrgemv<half, half, half, float>(
     blas::Op trans,
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
void csrgemv<half, half, float>(
     blas::Op trans,
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
void csrgemv<half, float, half>(
     blas::Op trans,
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
void csrgemv<half, float, float>(
     blas::Op trans,
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
void csrgemv<float, float, float>(
     blas::Op trans,
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
void csrgemv<float, float, float, double>(
     blas::Op trans,
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
void csrgemv<float, float, double>(
     blas::Op trans,
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
void csrgemv<float, double, float>(
     blas::Op trans,
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
void csrgemv<float, double, double>(
     blas::Op trans,
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
void csrgemv<double, double, double>(
     blas::Op trans,
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
void csrgemv<double, double, double, quadruple>(
     blas::Op trans,
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
void csrgemv<double, double, quadruple>(
     blas::Op trans,
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
void csrgemv<double, quadruple, double>(
     blas::Op trans,
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
void csrgemv<double, quadruple, quadruple>(
     blas::Op trans,
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
void csrgemv<quadruple, quadruple, quadruple>(
     blas::Op trans,
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
void csrgemv<quadruple, quadruple, quadruple, octuple>(
     blas::Op trans,
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
void csrgemv<quadruple, quadruple, octuple>(
     blas::Op trans,
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
void csrgemv<quadruple, octuple, quadruple>(
     blas::Op trans,
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
void csrgemv<quadruple, octuple, octuple>(
     blas::Op trans,
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
void csrgemv<octuple, octuple, octuple>(
     blas::Op trans,
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
void csrgemv<std::complex<half>, std::complex<half>, std::complex<half> >(
     blas::Op trans,
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
void csrgemv<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Op trans,
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
void csrgemv<std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Op trans,
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
void csrgemv<std::complex<half>, std::complex<float>, std::complex<half> >(
     blas::Op trans,
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
void csrgemv<std::complex<half>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
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
void csrgemv<std::complex<float>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
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
void csrgemv<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Op trans,
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
void csrgemv<std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Op trans,
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
void csrgemv<std::complex<float>, std::complex<double>, std::complex<float> >(
     blas::Op trans,
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
void csrgemv<std::complex<float>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
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
void csrgemv<std::complex<double>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
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
void csrgemv<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Op trans,
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
void csrgemv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Op trans,
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
void csrgemv<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
     blas::Op trans,
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
void csrgemv<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Op trans,
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
void csrgemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Op trans,
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
void csrgemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Op trans,
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
void csrgemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Op trans,
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
void csrgemv<std::complex<quadruple>, std::complex<octuple>, std::complex<quadruple> >(
     blas::Op trans,
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
void csrgemv<std::complex<quadruple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Op trans,
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
void csrgemv<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Op trans,
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

