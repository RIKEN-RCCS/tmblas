//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRGEMM_CPP
# define _CSRGEMM_CPP

#include "tmblas.hpp"
#include "csrgemm_tmpl.hpp"


namespace tmblas{
//real
//  HHH
template
void csrgemm<half, half, half>(
     blas::Op trans,
     idx_int m,
     idx_int n,     
     half const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     half const *x,
     idx_int ldx,
     half const &beta,
     half *y,
     idx_int ldy,
     half *w);
//  HHHF
template
void csrgemm<half, half, half, float>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     half const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     half const *x,
     idx_int ldx,
     half const &beta,
     half *y,
     idx_int ldy,
     float *w);
//  HHF ?
template
void csrgemm<half, half, float>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     float const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     half const *x,
     idx_int ldx,
     float const &beta,
     float *y,
     idx_int ldy,
     float *w);
//  HFH ?
template
void csrgemm<half, float, half>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     float const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     idx_int ldx,
     float const &beta,
     half *y,
     idx_int ldy,
     float *w);
// HFF
template
void csrgemm<half, float, float>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     float const &alpha,     
     half const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     idx_int ldx,
     float const &beta,
     float *y,
     idx_int ldy,
     float *w);

//  FFF
template
void csrgemm<float, float, float>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     float const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     idx_int ldx,
     float const &beta,
     float *y,
     idx_int ldy,
     float *w);
//  FFFD
template
void csrgemm<float, float, float, double>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     float const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     idx_int ldx,
     float const &beta,
     float *y,
     idx_int ldy,
     double *w);
//  FFD ?
template
void csrgemm<float, float, double>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     double const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     float const *x,
     idx_int ldx,
     double const &beta,
     double *y,
     idx_int ldy,
     double *w);
//  FDF ?
template
void csrgemm<float, double, float>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     double const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     idx_int ldx,
     double const &beta,
     float *y,
     idx_int ldy,
     double *w);
// FDD
template
void csrgemm<float, double, double>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     double const &alpha,     
     float const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     idx_int ldx,
     double const &beta,
     double *y,
     idx_int ldy,
     double *w);

//  DDD
template
void csrgemm<double, double, double>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     double const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     idx_int ldx,
     double const &beta,
     double *y,
     idx_int ldy,
     double *w);
//  DDDQ
template
void csrgemm<double, double, double, quadruple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     double const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     idx_int ldx,
     double const &beta,
     double *y,
     idx_int ldy,
     quadruple *w);
//  DDQ ?
template
void csrgemm<double, double, quadruple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     quadruple const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     double const *x,
     idx_int ldx,
     quadruple const &beta,
     quadruple *y,
     idx_int ldy,
     quadruple *w);
//  DQD ?
template
void csrgemm<double, quadruple, double>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     quadruple const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     idx_int ldx,
     quadruple const &beta,
     double *y,
     idx_int ldy,
     quadruple *w);
// DQQ
template
void csrgemm<double, quadruple, quadruple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     quadruple const &alpha,     
     double const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     idx_int ldx,
     quadruple const &beta,
     quadruple*y,
     idx_int ldy,
     quadruple *w);

//  QQQ
template
void csrgemm<quadruple, quadruple, quadruple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     quadruple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     idx_int ldx,
     quadruple const &beta,
     quadruple *y,
     idx_int ldy,
     quadruple *w);
//  QQQO
template
void csrgemm<quadruple, quadruple, quadruple, octuple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     quadruple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     idx_int ldx,
     quadruple const &beta,
     quadruple *y,
     idx_int ldy,
     octuple *w);
//  QQO ?
template
void csrgemm<quadruple, quadruple, octuple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     octuple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     quadruple const *x,
     idx_int ldx,
     octuple const &beta,
     octuple *y,
     idx_int ldy,
     octuple *w);
//  QOQ ?
template
void csrgemm<quadruple, octuple, quadruple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     octuple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     octuple const *x,
     idx_int ldx,
     octuple const &beta,
     quadruple *y,
     idx_int ldy,
     octuple *w);
// QOO
template
void csrgemm<quadruple, octuple, octuple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     octuple const &alpha,     
     quadruple const *av,
     idx_int const *ai,
     idx_int const *aj,
     octuple const *x,
     idx_int ldx,
     octuple const &beta,
     octuple*y,
     idx_int ldy,
     octuple *w);
// OOO
template
void csrgemm<octuple, octuple, octuple>(
     blas::Op trans,
     idx_int m,
     idx_int n,
     octuple const &alpha,     
     octuple const *av,
     idx_int const *ai,
     idx_int const *aj,
     octuple const *x,
     idx_int ldx,
     octuple const &beta,
     octuple *y,
     idx_int ldy,
     octuple *w);

//complex
//  HHH
template
void csrgemm<std::complex<half>, std::complex<half>, std::complex<half> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<half> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<half> const *x,
     idx_int ldx,
     std::complex<half> const &beta,
     std::complex<half> *y,
     idx_int ldy,
     std::complex<half> *w);
//  HHHF
template
void csrgemm<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<half> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<half> const *x,
     idx_int ldx,
     std::complex<half> const &beta,
     std::complex<half> *y,
     idx_int ldy,
     std::complex<float> *w);
//  HHF ?
template
void csrgemm<std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<float> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<half> const *x,
     idx_int ldx,
     std::complex<float> const &beta,
     std::complex<float> *y,
     idx_int ldy,
     std::complex<float> *w);
//  HFH ?
template
void csrgemm<std::complex<half>, std::complex<float>, std::complex<half> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<float> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     idx_int ldx,
     std::complex<float> const &beta,
     std::complex<half> *y,
     idx_int ldy,
     std::complex<float> *w);
// HFF
template
void csrgemm<std::complex<half>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<float> const &alpha,     
     std::complex<half> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     idx_int ldx,
     std::complex<float> const &beta,
     std::complex<float> *y,
     idx_int ldy,
     std::complex<float> *w);

//  FFF
template
void csrgemm<std::complex<float>, std::complex<float>, std::complex<float> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<float> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     idx_int ldx,
     std::complex<float> const &beta,
     std::complex<float> *y,
     idx_int ldy,
     std::complex<float> *w);
//  FFFD
template
void csrgemm<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<float> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     idx_int ldx,
     std::complex<float> const &beta,
     std::complex<float> *y,
     idx_int ldy,
     std::complex<double> *w);
//  FFD ?
template
void csrgemm<std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<double> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<float> const *x,
     idx_int ldx,
     std::complex<double> const &beta,
     std::complex<double> *y,
     idx_int ldy,
     std::complex<double> *w);
//  FDF ?
template
void csrgemm<std::complex<float>, std::complex<double>, std::complex<float> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<double> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     idx_int ldx,
     std::complex<double> const &beta,
     std::complex<float> *y,
     idx_int ldy,
     std::complex<double> *w);
// FDD
template
void csrgemm<std::complex<float>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<double> const &alpha,     
     std::complex<float> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     idx_int ldx,
     std::complex<double> const &beta,
     std::complex<double> *y,
     idx_int ldy,
     std::complex<double> *w);

//  DDD
template
void csrgemm<std::complex<double>, std::complex<double>, std::complex<double> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<double> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     idx_int ldx,
     std::complex<double> const &beta,
     std::complex<double> *y,
     idx_int ldy,
     std::complex<double> *w);
//  DDDQ
template
void csrgemm<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<double> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     idx_int ldx,
     std::complex<double> const &beta,
     std::complex<double> *y,
     idx_int ldy,
     std::complex<quadruple> *w);
//  DDQ ?
template
void csrgemm<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<quadruple> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<double> const *x,
     idx_int ldx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     idx_int ldy,
     std::complex<quadruple> *w);
//  DQD ?
template
void csrgemm<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<quadruple> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     idx_int ldx,
     std::complex<quadruple> const &beta,
     std::complex<double> *y,
     idx_int ldy,
     std::complex<quadruple> *w);
// DQQ
template
void csrgemm<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<quadruple> const &alpha,     
     std::complex<double> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     idx_int ldx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     idx_int ldy,
     std::complex<quadruple> *w);

//  QQQ
template
void csrgemm<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<quadruple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     idx_int ldx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     idx_int ldy,
     std::complex<quadruple> *w);
//  QQQO
template
void csrgemm<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<quadruple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     idx_int ldx,
     std::complex<quadruple> const &beta,
     std::complex<quadruple> *y,
     idx_int ldy,
     std::complex<octuple> *w);
//  QQO ?
template
void csrgemm<std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<octuple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<quadruple> const *x,
     idx_int ldx,
     std::complex<octuple> const &beta,
     std::complex<octuple> *y,
     idx_int ldy,
     std::complex<octuple> *w);
//  QOQ ?
template
void csrgemm<std::complex<quadruple>, std::complex<octuple>, std::complex<quadruple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<octuple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<octuple> const *x,
     idx_int ldx,
     std::complex<octuple> const &beta,
     std::complex<quadruple> *y,
     idx_int ldy,
     std::complex<octuple> *w);
// QOO
template
void csrgemm<std::complex<quadruple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<octuple> const &alpha,     
     std::complex<quadruple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<octuple> const *x,
     idx_int ldx,
     std::complex<octuple> const &beta,
     std::complex<octuple>*y,
     idx_int ldy,
     std::complex<octuple> *w);
// OOO
template
void csrgemm<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Op trans,
     idx_int m,
     idx_int n,
     std::complex<octuple> const &alpha,     
     std::complex<octuple> const *av,
     idx_int const *ai,
     idx_int const *aj,
     std::complex<octuple> const *x,
     idx_int ldx,
     std::complex<octuple> const &beta,
     std::complex<octuple> *y,
     idx_int ldy,
     std::complex<octuple> *w);
  
  
} // namespace tmblas
#endif

