//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRHEMV_CPP
# define _CSRHEMV_CPP

#include "tmblas.hpp"
#include "csrhemv_tmpl.hpp"


namespace tmblas{
//complex
//  HHH
template
void csrhemv<std::complex<half>, std::complex<half>, std::complex<half> >(
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
void csrhemv<std::complex<half>, std::complex<half>, std::complex<half>, std::complex<float> >(
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
void csrhemv<std::complex<half>, std::complex<half>, std::complex<float> >(
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
void csrhemv<std::complex<half>, std::complex<float>, std::complex<half> >(
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
void csrhemv<std::complex<half>, std::complex<float>, std::complex<float> >(
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
void csrhemv<std::complex<float>, std::complex<float>, std::complex<float> >(
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
void csrhemv<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<double> >(
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
void csrhemv<std::complex<float>, std::complex<float>, std::complex<double> >(
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
void csrhemv<std::complex<float>, std::complex<double>, std::complex<float> >(
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
void csrhemv<std::complex<float>, std::complex<double>, std::complex<double> >(
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
void csrhemv<std::complex<double>, std::complex<double>, std::complex<double> >(
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
void csrhemv<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<quadruple> >(
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
void csrhemv<std::complex<double>, std::complex<double>, std::complex<quadruple> >(
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
void csrhemv<std::complex<double>, std::complex<quadruple>, std::complex<double> >(
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
void csrhemv<std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >(
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
void csrhemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(
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
void csrhemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
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
void csrhemv<std::complex<quadruple>, std::complex<quadruple>, std::complex<octuple> >(
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
void csrhemv<std::complex<quadruple>, std::complex<octuple>, std::complex<quadruple> >(
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
void csrhemv<std::complex<quadruple>, std::complex<octuple>, std::complex<octuple> >(
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
void csrhemv<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(
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
#endif

