//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRSYMM_CPP
# define _CSRSYMM_CPP

#include "tmblas.hpp"
#include "csrsymm_tmpl.hpp"


namespace tmblas{
//real
//  HHH
template
void csrsymm<half, half, half>(
     blas::Uplo uplo,
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
void csrsymm<half, half, half, float>(
     blas::Uplo uplo,
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
void csrsymm<half, half, float>(
     blas::Uplo uplo,
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
void csrsymm<half, float, half>(
     blas::Uplo uplo,
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
void csrsymm<half, float, float>(
     blas::Uplo uplo,
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
void csrsymm<float, float, float>(
     blas::Uplo uplo,
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
void csrsymm<float, float, float, double>(
     blas::Uplo uplo,
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
void csrsymm<float, float, double>(
     blas::Uplo uplo,
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
void csrsymm<float, double, float>(
     blas::Uplo uplo,
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
  void csrsymm<float, double, double>(
     blas::Uplo uplo,
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
void csrsymm<double, double, double>(
     blas::Uplo uplo,
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
void csrsymm<double, double, double, quadruple>(
     blas::Uplo uplo,
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
void csrsymm<double, double, quadruple>(
     blas::Uplo uplo,
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
void csrsymm<double, quadruple, double>(
     blas::Uplo uplo,
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
void csrsymm<double, quadruple, quadruple>(
     blas::Uplo uplo,
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
void csrsymm<quadruple, quadruple, quadruple>(
     blas::Uplo uplo,
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
void csrsymm<quadruple, quadruple, quadruple, octuple>(
     blas::Uplo uplo,
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
void csrsymm<quadruple, quadruple, octuple>(
     blas::Uplo uplo,
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
void csrsymm<quadruple, octuple, quadruple>(
     blas::Uplo uplo,
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
void csrsymm<quadruple, octuple, octuple>(
     blas::Uplo uplo,
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
void csrsymm<octuple, octuple, octuple>(
     blas::Uplo uplo,
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
void csrsymm<std::complex<half>, std::complex<half>, std::complex<half> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<half>, std::complex<half>, std::complex<float> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<half>, std::complex<float>, std::complex<half> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<half>, std::complex<float>, std::complex<float> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<float>, std::complex<float>, std::complex<float> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<float>, std::complex<float>, std::complex<double> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<float>, std::complex<double>, std::complex<float> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<float>, std::complex<double>, std::complex<double> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<double>, std::complex<double>, std::complex<double> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<quadruple>, std::complex<octuple>, std::complex<quadruple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<quadruple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Uplo uplo,
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
void csrsymm<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
     blas::Uplo uplo,
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

