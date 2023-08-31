//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblasarch.hpp"
#include "mixedp_type_conv_tmpl.hpp"

namespace tmblas {
// HH
template
half type_conv<half, half>(half const &b);
// HF
template
half type_conv<half, float>(float const &b);
// FH
template
float type_conv<float, half>(half const &b);
// FF
template
float type_conv<float, float>(float const &b);

// FD
template
float type_conv<float, double>(double const &b);
// DF
template
double type_conv<double, float>(float const &b);
// DD  
template
double type_conv<double, double>(double const &b);

// QQ
template
quadruple type_conv<quadruple, quadruple>(quadruple const &b);


//OO
template
octuple type_conv<octuple, octuple>(octuple const &b);

// HI
template
half type_conv<half, int>(int const &b);

// FI
template
float type_conv<float, int>(int const &b);

// DI
template
double type_conv<double, int>(int const &b);
  
// QI
template
quadruple type_conv<quadruple, int>(int const &b);

//OI
template
octuple type_conv<octuple, int>(int const &b);

  
#ifdef QD_LIBRARY
// DQ
template<>
double type_conv<double, dd_real>(dd_real const &b) {
  return to_double(b);
}
// QD
template
dd_real type_conv<dd_real, double>(double const &b);


// DO
template<>
double type_conv<double, qd_real>(qd_real const &b) {
  return to_double(b);
}
// OD
template
qd_real type_conv<qd_real, double>(double const &b);


// QO
template<>
dd_real type_conv<dd_real, qd_real>(qd_real const &b) {
  return to_dd_real(b);
}
// OQ
template
qd_real type_conv<qd_real, dd_real>(dd_real const &b);
  
#elif defined(MPFR)
// DQ downcast
template<>
double type_conv<double, mpfr128>(mpfr128 const &b) {
  return mpfr_get_d(b._x, mpfrrnd);
}
// DO downcast
template<>
double type_conv<double, mpfr256>(mpfr256 const &b) {
  return mpfr_get_d(b._x, mpfrrnd);
}
// QD upcast
template<>
mpfr128 type_conv<mpfr128, double>(double const &b) {
  mpfr128 a;  
  mpfr_set_d(a._x, b, mpfrrnd);
  return a;
}
// OD upcast
template<>
mpfr256 type_conv<mpfr256, double>(double const &b) {
  mpfr256 a;  
  mpfr_set_d(a._x, b, mpfrrnd);
  return a;
}
// QO downcast
template<>
mpfr128 type_conv<mpfr128, mpfr256>(mpfr256 const &b) {
  mpfr128 a;  
  mpfr_set(a._x, b._x, mpfrrnd);
  return a;
}
// OQ upcast
template<>
mpfr256 type_conv<mpfr256, mpfr128>(mpfr128 const &b) {
  mpfr256 a;
  mpfr_set(a._x, b._x, mpfrrnd);
  return a;
}
#else
// DQ
template
double type_conv<double, quadruple>(quadruple const &b);

// QD
template
quadruple type_conv<quadruple, double>(double const &b);

// DO
template
double type_conv<double, octuple>(octuple const &b);

// OD
template
octuple type_conv<octuple, double>(double const &b);


// QO
template
quadruple type_conv<quadruple, octuple>(octuplex const &b);
#endif

// complex
// HH
template
std::complex<half> type_conv<std::complex<half>, std::complex<half> >(std::complex<half> const &b);

  // HHr
template
std::complex<half> type_conv<std::complex<half>, half>(half const &b);

// HF
template
std::complex<half> type_conv<std::complex<half>, std::complex<float> >(std::complex<float> const &b);

// HFr
template
std::complex<half> type_conv<std::complex<half>, float>(float const &b);

// FH
template
std::complex<float> type_conv<std::complex<float>, std::complex<half> >(std::complex<half> const &b);

// FHr
template
std::complex<float> type_conv<std::complex<float>, half>(half const &b);

// FF
template
std::complex<float> type_conv<std::complex<float>, std::complex<float> >(std::complex<float> const &b);

  // FFr
template
std::complex<float> type_conv<std::complex<float>, float>(float const &b);

// FD
template
std::complex<float> type_conv<std::complex<float>, std::complex<double> >(std::complex<double> const &b);

// FDr
template
std::complex<float> type_conv<std::complex<float>, double>(double const &b);

// DF
template
std::complex<double> type_conv<std::complex<double>, std::complex<float> >(std::complex<float> const &b);
// DFr
template
std::complex<double> type_conv<std::complex<double>, float>(float const &b);

// DD  
template
std::complex<double> type_conv<std::complex<double>, std::complex<double> >(std::complex<double> const &b);

// DDr
template
std::complex<double> type_conv<std::complex<double>, double>(double const &b);

// QQ
template
std::complex<quadruple> type_conv<std::complex<quadruple>, std::complex<quadruple> >(std::complex<quadruple> const &b);

// QQr
template
std::complex<quadruple> type_conv<std::complex<quadruple>, quadruple>(quadruple const &b);

//OO
template
std::complex<octuple> type_conv<std::complex<octuple>, std::complex<octuple> >(std::complex<octuple> const &b);

//OOr
template
std::complex<octuple> type_conv<std::complex<octuple>, octuple>(octuple const &b);
  
// complex
// DQ downcast
template
std::complex<double> type_conv<std::complex<double>, std::complex<quadruple> >(std::complex<quadruple> const &b);

// DO downcast
template
std::complex<double> type_conv<std::complex<double>, std::complex<octuple> >(std::complex<octuple> const &b);

// DQr downcast
template
std::complex<double> type_conv<std::complex<double>, quadruple>(quadruple const &b);

// DOr downcast
template
std::complex<double> type_conv<std::complex<double>, octuple>(octuple const &b);

// QD upcast
template
std::complex<quadruple> type_conv<std::complex<quadruple>, std::complex<double> >(std::complex<double> const &b);

// OD upcast
template
std::complex<octuple> type_conv<std::complex<octuple>, std::complex<double> >(std::complex<double> const &b);

// QDr upcast
template
std::complex<quadruple> type_conv<std::complex<quadruple>, double>(double const &b);

// QO downcast
template
std::complex<quadruple> type_conv<std::complex<quadruple>, std::complex<octuple> >(std::complex<octuple> const &b);

// QOr downcast
template
std::complex<quadruple> type_conv<std::complex<quadruple>, octuple>(octuple const &b);

// OQ upcast
template
std::complex<octuple> type_conv<std::complex<octuple>, std::complex<quadruple> >(std::complex<quadruple> const &b);

// OQr upcast
template
std::complex<octuple> type_conv<std::complex<octuple>, quadruple>(quadruple const &b);
  
} // namespace

