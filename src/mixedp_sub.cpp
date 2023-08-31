//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblasarch.hpp"
#include "mixedp_sub_tmpl.hpp"

namespace tmblas {
// instantiations
// real
// HHH
template
void mixedp_sub<half, half, half>(half &c, half const &a, half const &b);
// FHH
template
void mixedp_sub<float, half, half>(float &c, half const &a, half const &b);
// FFH
template
void mixedp_sub<float, float, half>(float &c, float const &a, half const &b);
// FHF
template
void mixedp_sub<float, half, float>(float &c, half const &a, float const &b);
// FFF
template
void mixedp_sub<float, float, float>(float &c, float const &a, float const &b);
// DFF
template
void mixedp_sub<double, float, float>(double &c, float const &a, float const &b);
// DDF
template
void mixedp_sub<double, double, float>(double &c, double const &a, float const &b);
template
// DFD
void mixedp_sub<double, float, double>(double &c, float const &a, double const &b);
// DDD
template
void mixedp_sub<double, double, double>(double &c, double const &a, double const &b);

#ifdef QD_LIBRARY
// QDD
template<>
void mixedp_sub<dd_real, double, double>(dd_real &c, double const &a, double const &b)
{
  c = type_conv<dd_real, double>(a) - b; // dedicated function c = a - b with two double?
}

// QQD
template<>
void mixedp_sub<dd_real, dd_real, double>(dd_real &c, dd_real const &a, double const &b)
{
  c = a - b; // dedicated function c = a - b with two double?
}

// QDQ
template<>
void mixedp_sub<dd_real, double, dd_real>(dd_real &c, double const &a, dd_real const &b)
{
  c = a - b;
}

// QQQ
template
void mixedp_sub<dd_real, dd_real, dd_real>(dd_real &c, dd_real const &a, dd_real const &b);

// OQQ
template<>
void mixedp_sub<qd_real, dd_real, dd_real>(qd_real &c, dd_real const &a, dd_real const &b)
{
  c = type_conv<qd_real, dd_real>(a) - b; // dedicated function c = a - b with two dd_real?
}

// OOQ
template<>
void mixedp_sub<qd_real, qd_real, dd_real>(qd_real &c, qd_real const &a, dd_real const &b)
{
  c = a - b; // dedicated function c = a - b with two dd_real?
}

// OQO
template<>
void mixedp_sub<qd_real, dd_real, qd_real>(qd_real &c, dd_real const &a, qd_real const &b)
{
  c = a - b;
}

// OOO
template
void mixedp_sub<qd_real, qd_real, qd_real>(qd_real &c, qd_real const &a, qd_real const &b);
#elif defined(MPFR)
// QDD
template<>
void mixedp_sub<mpfr128, double, double>(mpfr128 &c, double const &a, double const &b) {
  mpfr128 d;
  mpfr_set_d(d._x, a, mpfrrnd);  // upcasting a
  mpfr_sub_d(c._x, d._x, b, mpfrrnd);
}

// QQD
template<>
void mixedp_sub<mpfr128, mpfr128, double>(mpfr128 &c, mpfr128 const &a, double const &b) {
  mpfr_sub_d(c._x, a._x, b, mpfrrnd);
}

// QDQ
template<>
void mixedp_sub<mpfr128, double, mpfr128>(mpfr128 &c, double const &a, mpfr128 const &b) {
  mpfr_d_sub(c._x, a, b._x, mpfrrnd);
}

// QQQ
template<>
void mixedp_sub<mpfr128, mpfr128, mpfr128 >(mpfr128 &c, mpfr128 const &a, mpfr128 const &b)
{
  mpfr_sub(c._x, a._x, b._x, mpfrrnd);
}

// OQO
template<>
void mixedp_sub<mpfr256, mpfr128, mpfr256 >(mpfr256 &c, mpfr128 const &a,
		 mpfr256 const &b)
{
  mpfr_sub(c._x, a._x, b._x, mpfrrnd);
}

// OOQ
template<>
void mixedp_sub<mpfr256, mpfr256, mpfr128>(mpfr256 &c, mpfr256 const &a,
		 mpfr128 const &b)
{
  mpfr_sub(c._x, a._x, b._x, mpfrrnd);
}

// OQQ
template<>
void mixedp_sub<mpfr256, mpfr128, mpfr128>(mpfr256 &c, mpfr128 const &a,
		 mpfr128 const &b)
{
  mpfr256 worka = type_conv<mpfr128, mpfr128>(a);  
  mpfr_sub(c._x, worka._x, b._x, mpfrrnd);
}

// OOO
template<>
void mixedp_sub<mpfr256, mpfr256, mpfr256 >(mpfr256 &c, mpfr256 const &a, mpfr256 const &b) {
  mpfr_sub(c._x, a._x, b._x, mpfrrnd);
}

#else
// QDD
template
void mixedp_sub<quadruple, double, double>(quadruple &c, double const &a, double const &b);
// QQD
template
void mixedp_sub<quadruple, quadruple, double>(quadruple &c, quadruple const &a, double const &b);
// QDQ
template
void mixedp_sub<quadruple, double, quadruple>(quadruple &c, double const &a, quadruple const &b);

// QQQ
template
void mixedp_sub<quadruple, quadruple, quadruple>(quadruple &c, quadruple const &a, quadruple const &b);
// OQQ
template
void mixedp_sub<octuple, quadruple, quadruple>(octuple &c, quadruple const &a, quadruple const &b);
// OOQ
template
void mixedp_sub<octuple, octuple, quadruple>(octcuple &c, octuple const &a, quadruple const &b);
//OQO
template
void mixedp_sub<octuple, quadruple, octuple>(octuple &c, quadruple const &a, octuple const &b);
// OOO
template
void mixedp_sub<octuple, octuple, octuple>(octuple &c, octuple const &a, octuple const &b);
#endif
// complex
// HHH
template
void mixedp_sub<std::complex<half>, std::complex<half>, std::complex<half> >(std::complex<half> &c, std::complex<half> const &a, std::complex<half> const &b);

// HHHr
template
void mixedp_sub<std::complex<half>, std::complex<half>, half >(std::complex<half> &c, std::complex<half> const &a, half const &b);

// HHrH
template
void mixedp_sub<std::complex<half>, half, std::complex<half> >(std::complex<half> &c, half const &a, std::complex<half> const &b);

// HHrHr
template
void mixedp_sub<std::complex<half>, half, half>(std::complex<half> &c, half const &a, half const &b);

// FHH
template
void mixedp_sub<std::complex<float>, std::complex<half>, std::complex<half> >(std::complex<float> &c, std::complex<half> const &a, std::complex<half> const &b);
// FHHr
template
void mixedp_sub<std::complex<float>, std::complex<half>, half>(std::complex<float> &c, std::complex<half> const &a, half const &b);
// FHrH
template
void mixedp_sub<std::complex<float>, half, std::complex<half> >(std::complex<float> &c, half const &a, std::complex<half> const &b);
// FHrHr
template
void mixedp_sub<std::complex<float>, half, half>(std::complex<float> &c, half const &a, half const &b);

// FFH
template
void mixedp_sub<std::complex<float>, std::complex<float>, std::complex<half> >(std::complex<float> &c, std::complex<float> const &a, std::complex<half> const &b);
// FFHr
template
void mixedp_sub<std::complex<float>, std::complex<float>, half>(std::complex<float> &c, std::complex<float> const &a, half const &b);
// FFrH
template
void mixedp_sub<std::complex<float>, float, std::complex<half> >(std::complex<float> &c, float const &a, std::complex<half> const &b);
// FFrHr
template
void mixedp_sub<std::complex<float>, float, half>(std::complex<float> &c, float const &a, half const &b);

// FHF
template
void mixedp_sub<std::complex<float>, std::complex<half>, std::complex<float> >(std::complex<float> &c, std::complex<half> const &a, std::complex<float> const &b);
// FHFr
template
void mixedp_sub<std::complex<float>, std::complex<half>, float>(std::complex<float> &c, std::complex<half> const &a, float const &b);
// FHrF
template
void mixedp_sub<std::complex<float>, half, std::complex<float> >(std::complex<float> &c, half const &a, std::complex<float> const &b);
//FHrFr
template
void mixedp_sub<std::complex<float>, half, float>(std::complex<float> &c, half const &a, float const &b);

//FFF
template
void mixedp_sub<std::complex<float>, std::complex<float>, std::complex<float> >(std::complex<float> &c, std::complex<float> const &a, std::complex<float> const &b);

//FFFr
template
void mixedp_sub<std::complex<float>, std::complex<float>, float>(std::complex<float> &c, std::complex<float> const &a, float const &b);

//FFrF
template
void mixedp_sub<std::complex<float>, float, std::complex<float> >(std::complex<float> &c, float const &a, std::complex<float> const &b);

//FFrFr
template
void mixedp_sub<std::complex<float>, float, float>(std::complex<float> &c, float const &a, float const &b);

// DFF
template
void mixedp_sub<std::complex<double>, std::complex<float>, std::complex<float> >(std::complex<double> &c, std::complex<float> const &a, std::complex<float> const &b);
// DFFr
template
void mixedp_sub<std::complex<double>, std::complex<float>, float>(std::complex<double> &c, std::complex<float> const &a, float const &b);
// DFrF
template
void mixedp_sub<std::complex<double>, float, std::complex<float> >(std::complex<double> &c, float const &a, std::complex<float> const &b);
//DFrFr
template
void mixedp_sub<std::complex<double>, float, float>(std::complex<double> &c, float const &a, float const &b);

// DDF
template
void mixedp_sub<std::complex<double>, std::complex<double>, std::complex<float> >(std::complex<double> &c, std::complex<double> const &a, std::complex<float> const &b);
// DDFr
template
void mixedp_sub<std::complex<double>, std::complex<double>, float>(std::complex<double> &c, std::complex<double> const &a, float const &b);
// DDrF
template
void mixedp_sub<std::complex<double>, double, std::complex<float> >(std::complex<double> &c, double const &a, std::complex<float> const &b);
// DDrFr
template
void mixedp_sub<std::complex<double>, double, float>(std::complex<double> &c, double const &a, float const &b);

// DFD
template
void mixedp_sub<std::complex<double>, std::complex<float>, std::complex<double> >(std::complex<double> &c, std::complex<float> const &a, std::complex<double> const &b);
// DFDr
template
void mixedp_sub<std::complex<double>, std::complex<float>, double>(std::complex<double> &c, std::complex<float> const &a, double const &b);
// DFrD
template
void mixedp_sub<std::complex<double>, float, std::complex<double> >(std::complex<double> &c, float const &a, std::complex<double> const &b);
// DFrDr
template
void mixedp_sub<std::complex<double>, float, double>(std::complex<double> &c, float const &a, double const &b);

// DDD
template
void mixedp_sub<std::complex<double>, std::complex<double>, std::complex<double> >(std::complex<double> &c, std::complex<double> const &a, std::complex<double> const &b);
// DDDr
template
void mixedp_sub<std::complex<double>, std::complex<double>, double>(std::complex<double> &c, std::complex<double> const &a, double const &b);
// DDrD
template
void mixedp_sub<std::complex<double>, double, std::complex<double> >(std::complex<double> &c, double const &a, std::complex<double> const &b);
//DDrDr
template
void mixedp_sub<std::complex<double>, double, double>(std::complex<double> &c, double const &a, double const &b);
//
// QDD
template
void mixedp_sub<std::complex<quadruple>, std::complex<double>, std::complex<double> >(std::complex<quadruple> &c, std::complex<double> const &a, std::complex<double> const &b);
// QDDr
template
void mixedp_sub<std::complex<quadruple>, std::complex<double>, double>(std::complex<quadruple> &c, std::complex<double> const &a, double const &b);
// QDrD
template
void mixedp_sub<std::complex<quadruple>, double, std::complex<double> >(std::complex<quadruple> &c, double const &a, std::complex<double> const &b);
// QDrDr
template
void mixedp_sub<std::complex<quadruple>, double, double >(std::complex<quadruple> &c, double const &a, double const &b);

// QQD
template
void mixedp_sub<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(std::complex<quadruple> &c, std::complex<quadruple> const &a, std::complex<double> const &b);
// QQDr
template
void mixedp_sub<std::complex<quadruple>, std::complex<quadruple>, double>(std::complex<quadruple> &c, std::complex<quadruple> const &a, double const &b);
// QQrD
template
void mixedp_sub<std::complex<quadruple>, quadruple, std::complex<double> >(std::complex<quadruple> &c, quadruple const &a, std::complex<double> const &b);
// QQrDr
template
void mixedp_sub<std::complex<quadruple>, quadruple, double>(std::complex<quadruple> &c, quadruple const &a, double const &b);

// QDQ
template
void mixedp_sub<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(std::complex<quadruple> &c, std::complex<double> const &a, std::complex<quadruple> const &b);
// QDQr
template
void mixedp_sub<std::complex<quadruple>, std::complex<double>, quadruple>(std::complex<quadruple> &c, std::complex<double> const &a, quadruple const &b);
// QDrQ
template
void mixedp_sub<std::complex<quadruple>, double, std::complex<quadruple> >(std::complex<quadruple> &c, double const &a, std::complex<quadruple> const &b);
// QDrQr
template
void mixedp_sub<std::complex<quadruple>, double, quadruple>(std::complex<quadruple> &c, double const &a, quadruple const &b);

// QQQ 
template
void mixedp_sub<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(std::complex<quadruple> &c, std::complex<quadruple> const &a, std::complex<quadruple> const &b);
// QQQr
template
void mixedp_sub<std::complex<quadruple>, std::complex<quadruple>, quadruple>(std::complex<quadruple> &c, std::complex<quadruple> const &a, quadruple const &b);
// QQrQ
template
void mixedp_sub<std::complex<quadruple>, quadruple, std::complex<quadruple> >(std::complex<quadruple> &c, quadruple const &a, std::complex<quadruple> const &b);
// QQrQr
template
void mixedp_sub<std::complex<quadruple>, quadruple, quadruple>(std::complex<quadruple> &c, quadruple const &a, quadruple const &b);
// OQQ
template
void mixedp_sub<std::complex<octuple>, std::complex<quadruple>, std::complex<quadruple> >(std::complex<octuple> &c, std::complex<quadruple> const &a, std::complex<quadruple> const &b);
// OQQr
template
void mixedp_sub<std::complex<octuple>, std::complex<quadruple>, quadruple>(std::complex<octuple> &c, std::complex<quadruple> const &a, quadruple const &b);
// OQrQ
template
void mixedp_sub<std::complex<octuple>, quadruple, std::complex<quadruple> >(std::complex<octuple> &c, quadruple const &a, std::complex<quadruple> const &b);
// OQrQr
template
void mixedp_sub<std::complex<octuple>, quadruple, quadruple>(std::complex<octuple> &c, quadruple const &a, quadruple const &b);


// OOQ
template
void mixedp_sub<std::complex<octuple>, std::complex<octuple>, std::complex<quadruple> >(std::complex<octuple> &c, std::complex<octuple> const &a, std::complex<quadruple> const &b);
// OOQr
template
void mixedp_sub<std::complex<octuple>, std::complex<octuple>, quadruple>(std::complex<octuple> &c, std::complex<octuple> const &a, quadruple const &b);
// OOrQ
template
void mixedp_sub<std::complex<octuple>, octuple, std::complex<quadruple> >(std::complex<octuple> &c, octuple const &a, std::complex<quadruple> const &b);
// OOrQr
template
void mixedp_sub<std::complex<octuple>, octuple, quadruple>(std::complex<octuple> &c, octuple const &a, quadruple const &b);

// OQO
template
void mixedp_sub<std::complex<octuple>, std::complex<quadruple>, std::complex<octuple> >(std::complex<octuple> &c, std::complex<quadruple> const &a, std::complex<octuple> const &b);
// OQOr
template
void mixedp_sub<std::complex<octuple>, std::complex<quadruple>, octuple>(std::complex<octuple> &c, std::complex<quadruple> const &a, octuple const &b);
// OQrO
template
void mixedp_sub<std::complex<octuple>, quadruple, std::complex<octuple> >(std::complex<octuple> &c, quadruple const &a, std::complex<octuple> const &b);
// OQrOr
template
void mixedp_sub<std::complex<octuple>, quadruple, octuple>(std::complex<octuple> &c, quadruple const &a, octuple const &b);

// OOO
template
void mixedp_sub<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(std::complex<octuple> &c, std::complex<octuple> const &a, std::complex<octuple> const &b);
// OOOr
template
void mixedp_sub<std::complex<octuple>, std::complex<octuple>, octuple>(std::complex<octuple> &c, std::complex<octuple> const &a, octuple const &b);
// OOrO
template
void mixedp_sub<std::complex<octuple>, octuple, std::complex<octuple> >(std::complex<octuple> &c, octuple const &a, std::complex<octuple> const &b);
// OOrOr
template
void mixedp_sub<std::complex<octuple>, octuple, octuple>(std::complex<octuple> &c, octuple const &a, octuple const &b);
//
} // namespace tmblas
