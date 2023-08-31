//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblasarch.hpp"
#include "mixedp_madd_tmpl.hpp"

namespace tmblas {
// instantiations
// real
// HHH
template
void mixedp_madd<half, half, half>(half &c, half const &a, half const &b);
// FHH
template
void mixedp_madd<float, half, half>(float &c, half const &a, half const &b);
// FFH
template
void mixedp_madd<float, float, half>(float &c, float const &a, half const &b);
// FHF
template
void mixedp_madd<float, half, float>(float &c, half const &a, float const &b);
// FFF
template
void mixedp_madd<float, float, float>(float &c, float const &a, float const &b);
// DFF
template
void mixedp_madd<double, float, float>(double &c, float const &a, float const &b);
// DDF
template
void mixedp_madd<double, double, float>(double &c, double const &a, float const &b);
template
// DFD
void mixedp_madd<double, float, double>(double &c, float const &a, double const &b);
// DDD
template
void mixedp_madd<double, double, double>(double &c, double const &a, double const &b);
//
  
#ifdef QD_LIBRARY
// QDD
template<>
void mixedp_madd<quadruple, double, double>(quadruple &c, double const &a, double const &b)
{
  c += type_conv<quadruple, double>(a) * b; // dedicated function c = a * b with two double?
}

// QQD
template<>
void mixedp_madd<quadruple, quadruple, double>(quadruple &c, quadruple const &a, double const &b)
{
  c += a * b; // dedicated function c = a * b with two double?
}

// QDQ
template<>
void mixedp_madd<quadruple, double, quadruple>(quadruple &c, double const &a, quadruple const &b)
{
  c += a * b;
}

// QQQ
template<>
void mixedp_madd<quadruple, quadruple, quadruple>(quadruple &c, quadruple const &a, quadruple const &b)
{
  madd<quadruple>(c, a, b);  // mono-precision
}

// OQQ
template<>
void mixedp_madd<octuple, quadruple, quadruple>(octuple &c, quadruple const &a, quadruple const &b)
{
  c += type_conv<octuple, quadruple>(a) * b; // dedicated function c = a * b with two quadruple?
}

// OOQ
template<>
void mixedp_madd<octuple, octuple, quadruple>(octuple &c, octuple const &a, quadruple const &b)
{
  c += a * b; // dedicated function c = a * b with two quadruple?
}

// OQO
template<>
void mixedp_madd<octuple, quadruple, octuple>(octuple &c, quadruple const &a, octuple const &b)
{
  c += a * b;
}

// OOO
template<>
void mixedp_madd<octuple, octuple, octuple>(octuple &c, octuple const &a, octuple const &b)
{
  madd<octuple>(c, a, b);  // mono-precision
}

#elif defined(MPFR)
// QDD
template<>
void mixedp_madd<mpfr128, double, double>(mpfr128 &c, double const &a, double const &b) {
  mpfr128 worka = type_conv<mpfr128, double>(a);
  mpfr128 workb = type_conv<mpfr128, double>(b);
  mpfr_fma(c._x, worka._x, workb._x, c._x, mpfrrnd);
}

// QQD
template<>
void mixedp_madd<mpfr128, mpfr128, double>(mpfr128 &c, mpfr128 const &a, double const &b) {
  mpfr128 workb = type_conv<mpfr128, double>(b);
  mpfr_fma(c._x, a._x, workb._x, c._x, mpfrrnd);
}

// QDQ
template<>
void mixedp_madd<mpfr128, double, mpfr128>(mpfr128 &c, double const &a, mpfr128 const &b) {
  mpfr128 worka = type_conv<mpfr128, double>(a);
  mpfr_fma(c._x, worka._x, b._x, c._x, mpfrrnd);
}

// QQQ
template<>
void mixedp_madd<mpfr128, mpfr128, mpfr128 >(mpfr128 &c, mpfr128 const &a, mpfr128 const &b) {
  mpfr_fma(c._x, a._x, b._x, c._x, mpfrrnd);
}

// OQO
template<>
void mixedp_madd<mpfr256, mpfr128, mpfr256 >(mpfr256 &c, mpfr128 const &a,
		 mpfr256 const &b)
{
  mpfr256 worka = type_conv<mpfr256, mpfr128>(a);
  mpfr_fma(c._x, worka._x, b._x, c._x, mpfrrnd);
}

// OOQ
template<>
void mixedp_madd<mpfr256, mpfr256, mpfr128>(mpfr256 &c, mpfr256 const &a,
		 mpfr128 const &b)
{
  mpfr256 workb = type_conv<mpfr256, mpfr128>(b);
  mpfr_fma(c._x, a._x, workb._x, c._x, mpfrrnd);
}

// OQQ
template<>
void mixedp_madd<mpfr256, mpfr128, mpfr128>(mpfr256 &c, mpfr128 const &a,
		 mpfr128 const &b)
{
  mpfr256 worka = type_conv<mpfr256, mpfr128>(a);  
  mpfr256 workb = type_conv<mpfr256, mpfr128>(b);
  mpfr_fma(c._x, worka._x, workb._x, c._x, mpfrrnd);
}

// OOO
template<>
void mixedp_madd<mpfr256, mpfr256, mpfr256 >(mpfr256 &c, mpfr256 const &a, mpfr256 const &b) {
  mpfr_fma(c._x, a._x, b._x, c._x, mpfrrnd);
}
#else
// QDD
template
void mixedp_madd<quadruple, double, double>(quadruple &c, double const &a, double const &b);
// QQD
template
void mixedp_madd<quadruple, quadruple, double>(quadruple &c, quadruple const &a, double const &b);
// QDQ
template
void mixedp_madd<quadruple, double, quadruple>(quadruple &c, double const &a, quadruple const &b);
// QQQ
template<>
void mixedp_madd<quadruple, quadruple, quadruple>(quadruple &c, quadruple const &a, quadruple const &b)
{
  madd<quadruple>(c, a, b); // mono-precision
}
//OQO
template
void mixedp_madd<octuple, quadruple, octuple>(octuple &c, quadruple const &a, octuple const &b);
// OOQ
template
void mixedp_madd<octuple, octuple, quadruple>(octcuple &c, octuple const &a, quadruple const &b);
// OQQ
template
void mixedp_madd<octuple, quadruple, quadruple>(octuple &c, quadruple const &a, quadruple const &b);
// OOO
template<>
void mixedp_madd<octuple, octuple, octuple>(octuple &c, octuple const &a, octuple const &b)
{
  madd<octuple>(c, a, b); // mono-precision
}
//
#endif

// complex
// HHH
template
void mixedp_madd<std::complex<half>, std::complex<half>, std::complex<half> >(std::complex<half> &c, std::complex<half> const &a, std::complex<half> const &b);
// HHHr
template
void mixedp_madd<std::complex<half>, std::complex<half>, half>(std::complex<half> &c, std::complex<half> const &a, half const &b);
// HHrH
template
void mixedp_madd<std::complex<half>, half, std::complex<half> >(std::complex<half> &c, half const &a, std::complex<half> const &b);
// HHrHr
template
void mixedp_madd<std::complex<half>, half, half>(std::complex<half> &c, half const &a, half const &b);
// FHH
template
void mixedp_madd<std::complex<float>, std::complex<half>, std::complex<half> >(std::complex<float> &c, std::complex<half> const &a, std::complex<half> const &b);
// FHHr
template
void mixedp_madd<std::complex<float>, std::complex<half>, half>(std::complex<float> &c, std::complex<half> const &a, half const &b);
// FHrH
template
void mixedp_madd<std::complex<float>, half, std::complex<half> >(std::complex<float> &c, half const &a, std::complex<half> const &b);
// FHrHr
template
void mixedp_madd<std::complex<float>, half, half>(std::complex<float> &c, half const &a, half const &b);
  
// FFH
template
void mixedp_madd<std::complex<float>, std::complex<float>, std::complex<half> >(std::complex<float> &c, std::complex<float> const &a, std::complex<half> const &b);
// FFHr
template
void mixedp_madd<std::complex<float>, std::complex<float>, half>(std::complex<float> &c, std::complex<float> const &a, half const &b);
// FFrH
template
void mixedp_madd<std::complex<float>, float, std::complex<half> >(std::complex<float> &c, float const &a, std::complex<half> const &b);
// FFrHr
template
void mixedp_madd<std::complex<float>, float, half>(std::complex<float> &c, float const &a, half const &b);


//FHF
template
void mixedp_madd<std::complex<float>, std::complex<half>, std::complex<float> >(std::complex<float> &c, std::complex<half> const &a, std::complex<float> const &b);
//FHFr
template
void mixedp_madd<std::complex<float>, std::complex<half>, float>(std::complex<float> &c, std::complex<half> const &a, float const &b);
//FHrF
template
void mixedp_madd<std::complex<float>, half, std::complex<float> >(std::complex<float> &c, half const &a, std::complex<float> const &b);
//FHrFr
template
void mixedp_madd<std::complex<float>, half, float>(std::complex<float> &c, half const &a, float const &b);

 //FFF
template
void mixedp_madd<std::complex<float>, std::complex<float>, std::complex<float> >(std::complex<float> &c, std::complex<float> const &a, std::complex<float> const &b);
//FFFr
template
void mixedp_madd<std::complex<float>, std::complex<float>, float>(std::complex<float> &c, std::complex<float> const &a, float const &b);
//FFrF
template
void mixedp_madd<std::complex<float>, float, std::complex<float> >(std::complex<float> &c, float const &a, std::complex<float> const &b);
//FFrFr
template
void mixedp_madd<std::complex<float>, float, float>(std::complex<float> &c, float const &a, float const &b);

//DFF
template
void mixedp_madd<std::complex<double>, std::complex<float>, std::complex<float> >(std::complex<double> &c, std::complex<float> const &a, std::complex<float> const &b);
//DFFr
template
void mixedp_madd<std::complex<double>, std::complex<float>, float>(std::complex<double> &c, std::complex<float> const &a, float const &b);
//DFrF
template
void mixedp_madd<std::complex<double>, float, std::complex<float> >(std::complex<double> &c, float const &a, std::complex<float> const &b);
//DFrFr
template
void mixedp_madd<std::complex<double>, float, float>(std::complex<double> &c, float const &a, float const &b);

// DDF
template
void mixedp_madd<std::complex<double>, std::complex<double>, std::complex<float> >(std::complex<double> &c, std::complex<double> const &a, std::complex<float> const &b);
// DDFr
template
void mixedp_madd<std::complex<double>, std::complex<double>, float>(std::complex<double> &c, std::complex<double> const &a, float const &b);
// DDrF
template
void mixedp_madd<std::complex<double>, double, std::complex<float> >(std::complex<double> &c, double const &a, std::complex<float> const &b);
// DDrFr
template
void mixedp_madd<std::complex<double>, double, float>(std::complex<double> &c, double const &a, float const &b);

// DFD
template
void mixedp_madd<std::complex<double>, std::complex<float>, std::complex<double> >(std::complex<double> &c, std::complex<float> const &a, std::complex<double> const &b);
// DFDr
template
void mixedp_madd<std::complex<double>, std::complex<float>, double>(std::complex<double> &c, std::complex<float> const &a, double const &b);
// DFrD
template
void mixedp_madd<std::complex<double>, float, std::complex<double> >(std::complex<double> &c, float const &a, std::complex<double> const &b);
// DFrDr
template
void mixedp_madd<std::complex<double>, float, double>(std::complex<double> &c, float const &a, double const &b);

//DDD
template
void mixedp_madd<std::complex<double>, std::complex<double>, std::complex<double> >(std::complex<double> &c, std::complex<double> const &a, std::complex<double> const &b);

//DDDr
template
void mixedp_madd<std::complex<double>, std::complex<double>, double>(std::complex<double> &c, std::complex<double> const &a, double const &b);

//DDrD
template
void mixedp_madd<std::complex<double>, double, std::complex<double> >(std::complex<double> &c, double const &a, std::complex<double> const &b);

//DDrDr
template
void mixedp_madd<std::complex<double>, double, double>(std::complex<double> &c, double const &a, double const &b);

// QDD
template
void mixedp_madd<std::complex<quadruple>, std::complex<double>, std::complex<double> >(std::complex<quadruple> &c, std::complex<double> const &a, std::complex<double> const &b);

// QDDr
template
void mixedp_madd<std::complex<quadruple>, std::complex<double>, double>(std::complex<quadruple> &c, std::complex<double> const &a, double const &b);

// QDrD
template
void mixedp_madd<std::complex<quadruple>, double, std::complex<double> >(std::complex<quadruple> &c, double const &a, std::complex<double> const &b);

// QDrDr
template
void mixedp_madd<std::complex<quadruple>, double, double>(std::complex<quadruple> &c, double const &a, double const &b);

// QQD
template
void mixedp_madd<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(std::complex<quadruple> &c, std::complex<quadruple> const &a, std::complex<double> const &b);

// QQDr
template
void mixedp_madd<std::complex<quadruple>, std::complex<quadruple>, double>(std::complex<quadruple> &c, std::complex<quadruple> const &a, double const &b);

// QQrD
template
void mixedp_madd<std::complex<quadruple>, quadruple, std::complex<double> > (std::complex<quadruple> &c, quadruple const &a, std::complex<double> const &b);

// QQrDr
template
void mixedp_madd<std::complex<quadruple>, quadruple, double> (std::complex<quadruple> &c, quadruple const &a, double const &b);

// QDQ
template
void mixedp_madd<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(std::complex<quadruple> &c, std::complex<double> const &a, std::complex<quadruple> const &b);

// QDQr
template
void mixedp_madd<std::complex<quadruple>, std::complex<double>, quadruple>(std::complex<quadruple> &c, std::complex<double> const &a, quadruple const &b);

// QDrQ
template
void mixedp_madd<std::complex<quadruple>, double, std::complex<quadruple> >(std::complex<quadruple> &c, double const &a, std::complex<quadruple> const &b);

// QDrQr
template
void mixedp_madd<std::complex<quadruple>, double, quadruple>(std::complex<quadruple> &c, double const &a, quadruple const &b);


// QQQ
template
void mixedp_madd<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(std::complex<quadruple> &c, std::complex<quadruple> const &a, std::complex<quadruple> const &b);
// QQQr
template
void mixedp_madd<std::complex<quadruple>, std::complex<quadruple>, quadruple>(std::complex<quadruple> &c, std::complex<quadruple> const &a, quadruple const &b);
// QQrQ
template
void mixedp_madd<std::complex<quadruple>, quadruple, std::complex<quadruple> >(std::complex<quadruple> &c, quadruple const &a, std::complex<quadruple> const &b);
// QQrQr
template
void mixedp_madd<std::complex<quadruple>, quadruple, quadruple >(std::complex<quadruple> &c, quadruple const &a, quadruple const &b);


// OQQ
template
void mixedp_madd<std::complex<octuple>, std::complex<quadruple>, std::complex<quadruple> >(std::complex<octuple> &c, std::complex<quadruple> const &a, std::complex<quadruple> const &b);
// OQQr
template
void mixedp_madd<std::complex<octuple>, std::complex<quadruple>, quadruple>(std::complex<octuple> &c, std::complex<quadruple> const &a, quadruple const &b);
// OQrQ
template
void mixedp_madd<std::complex<octuple>, quadruple, std::complex<quadruple> >(std::complex<octuple> &c, quadruple const &a, std::complex<quadruple> const &b);
// OQrQr
template
void mixedp_madd<std::complex<octuple>, quadruple, quadruple>(std::complex<octuple> &c, quadruple const &a, quadruple const &b);

// OOQ
template
void mixedp_madd<std::complex<octuple>, std::complex<octuple>, std::complex<quadruple> >(std::complex<octuple> &c, std::complex<octuple> const &a, std::complex<quadruple> const &b);

// OOQr
template
void mixedp_madd<std::complex<octuple>, std::complex<octuple>, quadruple>(std::complex<octuple> &c, std::complex<octuple> const &a, quadruple const &b);

// OOrQ
template
void mixedp_madd<std::complex<octuple>, octuple, std::complex<quadruple> >(std::complex<octuple> &c, octuple const &a, std::complex<quadruple> const &b);

// OQO
template
void mixedp_madd<std::complex<octuple>, std::complex<quadruple>, std::complex<octuple> >(std::complex<octuple> &c, std::complex<quadruple> const &a, std::complex<octuple> const &b);
// OQOr
template
void mixedp_madd<std::complex<octuple>, std::complex<quadruple>, octuple>(std::complex<octuple> &c, std::complex<quadruple> const &a, octuple const &b);
// OQrO
template
void mixedp_madd<std::complex<octuple>, quadruple, std::complex<octuple> >(std::complex<octuple> &c, quadruple const &a, std::complex<octuple> const &b);
// OQrOr
template
void mixedp_madd<std::complex<octuple>, quadruple, octuple>(std::complex<octuple> &c, quadruple const &a, octuple const &b);

// OOO
template
void mixedp_madd<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(std::complex<octuple> &c, std::complex<octuple> const &a, std::complex<octuple> const &b);
// OOOr
template
void mixedp_madd<std::complex<octuple>, std::complex<octuple>, octuple>(std::complex<octuple> &c, std::complex<octuple> const &a, octuple const &b);
// OOrO
template
void mixedp_madd<std::complex<octuple>, octuple, std::complex<octuple> >(std::complex<octuple> &c, octuple const &a, std::complex<octuple> const &b);
// OOrOr
template
void mixedp_madd<std::complex<octuple>, octuple, octuple>(std::complex<octuple> &c, octuple const &a, octuple const &b);

#if 0 // fmms/fmma in MPFR for future improvements
// QDD
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<double>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<double> const &a, std::complex<double> const &b)
{
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));
  mpfr128 workbr = type_conv<mpfr128, double>(real(b));
  mpfr128 workbi = type_conv<mpfr128, double>(imag(b));
  mpfr128 cr1(0);
  mpfr128 ci1(0);
  mpfr_fmms(cr1._x, workar._x, workbr._x, workai._x, workbi._x, mpfrrnd);
  mpfr128 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, workar._x, workbi._x, workai._x, workbr._x, mpfrrnd);
  mpfr128 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QDDr
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<double>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<double> const &a, double const &b)
{
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));
  mpfr128 workb = type_conv<mpfr128, double>(b);
  mpfr128 cr = real(c);
  mpfr128 ci = imag(c);
  mpfr_fma(cr._x, workar._x, workb._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, workai._x, workb._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}
// QDrD
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<double>, std::complex<double> >(std::complex<mpfr128> &c, double const &a, std::complex<double> const &b)
{
  mpfr128 worka = type_conv<mpfr128, double>(a);
  mpfr128 workbr = type_conv<mpfr128, double>(real(b));
  mpfr128 workbi = type_conv<mpfr128, double>(imag(b));
  mpfr128 cr = real(c);
  mpfr128 ci = imag(c);
  mpfr_fma(cr._x, worka._x, workbi._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, worka._x, workbi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// QQD
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, std::complex<double> const &b)
{
  mpfr128 workbr = type_conv<mpfr128, double>(real(b));
  mpfr128 workbi = type_conv<mpfr128, double>(imag(b));
  mpfr128 ar = real(a);
  mpfr128 ai = imag(a);
  mpfr128 cr1(0);
  mpfr128 ci1(0);
  mpfr_fmms(cr1._x, ar._x, workbr._x, ai._x, workbi._x, mpfrrnd);
  mpfr128 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, ar._x, workbi._x, ai._x, workbr._x, mpfrrnd);
  mpfr128 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QQDr
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, double const &b)
{
  mpfr128 workb = type_conv<mpfr128, double>(b);
  mpfr128 ar = real(a);
  mpfr128 ai = imag(a);
  mpfr128 cr(0);
  mpfr128 ci(0);

  mpfr_fma(cr._x, ar._x, workb._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, ai._x, workb._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}
// QQrD
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<double> >(std::complex<mpfr128> &c, mpfr128 const &a, std::complex<double> const &b)
{
  mpfr128 workbr = type_conv<mpfr128, double>(real(b));
  mpfr128 workbi = type_conv<mpfr128, double>(imag(b));  
  mpfr128 cr = real(c);
  mpfr128 ci = imag(c);
  mpfr_fma(cr._x, a._x, workbr._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, a._x, workbi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}
// QDQ
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<double>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<double> const &a, std::complex<mpfr128> const &b)
{
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));
  mpfr128 br = real(b);
  mpfr128 bi = imag(b);
  mpfr128 cr1(0);
  mpfr128 ci1(0);
  mpfr_fmms(cr1._x, workar._x, br._x, workai._x, bi._x, mpfrrnd);
  mpfr128 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, workar._x, bi._x, workai._x, br._x, mpfrrnd);
  mpfr128 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QDQr
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<double>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<double> const &a, mpfr128 const &b)
{
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));  
  mpfr128 cr = real(c);
  mpfr128 ci = imag(c);
  mpfr_fma(cr._x, workar._x, b._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, workai._x, b._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// QDrQ
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<double>, std::complex<mpfr128> >(std::complex<mpfr128> &c, double const &a, std::complex<mpfr128> const &b)
{
  mpfr128 worka = type_conv<mpfr128, double>(a);
  mpfr128 br = real(b);
  mpfr128 bi = imag(b);
  
  mpfr128 cr = real(c);
  mpfr128 ci = imag(c);
  mpfr_fma(cr._x, worka._x, br._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, worka._x, bi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// QQQ
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, std::complex<mpfr128> const &b)
{
  mpfr128 ar = real(a);
  mpfr128 ai = imag(a);
  mpfr128 br = real(b);
  mpfr128 bi = imag(b);
  mpfr128 cr1(0);
  mpfr128 ci1(0);
  mpfr_fmms(cr1._x, ar._x, br._x, ai._x, bi._x, mpfrrnd);
  mpfr128 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, ar._x, bi._x, ai._x, br._x, mpfrrnd);
  mpfr128 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QQQr
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, mpfr128 const &b)
{
  mpfr128 ar = real(a);
  mpfr128 ai = imag(a);

  mpfr128 cr = real(c);
  mpfr128 ci = imag(c);
  mpfr_fma(cr._x, ar._x, b._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, ai._x, b._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}
// QQrQ
template<>
void mixedp_madd<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr128> &c, mpfr128 const &a, std::complex<mpfr128> const &b)
{
  mpfr128 br = real(b);
  mpfr128 bi = imag(b);

  mpfr128 cr = real(c);
  mpfr128 ci = imag(c);
  mpfr_fma(cr._x, a._x, br._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, a._x, bi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// OQO
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 std::complex<mpfr256> const &b)
{
  mpfr256 workar = type_conv<mpfr256, mpfr128>(real(a));
  mpfr256 workai = type_conv<mpfr256, mpfr128>(imag(a));
  mpfr256 br = real(b);
  mpfr256 bi = imag(b);
  mpfr256 cr1(0);
  mpfr256 ci1(0);
  mpfr_fmms(cr1._x, workar._x, br._x, workai._x, bi._x, mpfrrnd);
  mpfr256 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, workar._x, bi._x, workai._x, br._x, mpfrrnd);
  mpfr256 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OQOr
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 mpfr256 const &b)
{
  mpfr128 ar = real(a);
  mpfr128 ai = imag(a);
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, ar._x, b._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, ai._x, b._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}
// OQrO
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr256> >(std::complex<mpfr256> &c, mpfr128 const &a,
		 std::complex<mpfr256> const &b)
{
  mpfr256 br = real(b);
  mpfr256 bi = imag(b);
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, a._x, br._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, a._x, bi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// OOQ
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr256 ar = real(a);
  mpfr256 ai = imag(a);
  mpfr256 workbr = type_conv<mpfr256, mpfr128>(real(b));
  mpfr256 workbi = type_conv<mpfr256, mpfr128>(imag(b));
  mpfr256 cr1(0);
  mpfr256 ci1(0);
  mpfr_fmms(cr1._x, ar._x, workbr._x, ai._x, workbi._x, mpfrrnd);
  mpfr256 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, ar._x, workbi._x, ai._x, workbr._x, mpfrrnd);
  mpfr256 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OOQr
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a,
		 mpfr128 const &b)
{
  mpfr256 ar = real(a);
  mpfr256 ai = imag(a);
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, ar._x, b._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, ai._x, b._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// OOrQ
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr128> >(std::complex<mpfr256> &c, mpfr256 const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr128 br = real(b);
  mpfr128 bi = imag(b);
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, a._x, br._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, a._x, bi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// OQQ
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr256 workar = type_conv<mpfr256, mpfr128>(real(a));
  mpfr256 workai = type_conv<mpfr256, mpfr128>(imag(a));
  mpfr256 workbr = type_conv<mpfr256, mpfr128>(real(b));
  mpfr256 workbi = type_conv<mpfr256, mpfr128>(imag(b));
  mpfr256 cr1(0);
  mpfr256 ci1(0);
  mpfr_fmms(cr1._x, workar._x, workbr._x, workai._x, workbi._x, mpfrrnd);
  mpfr256 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, workar._x, workbi._x, workai._x, workbr._x, mpfrrnd);
  mpfr256 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OQQr
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 mpfr128 const &b)
{
  mpfr256 workar = type_conv<mpfr256, mpfr128>(real(a));
  mpfr256 workai = type_conv<mpfr256, mpfr128>(imag(a));
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, workar._x, b._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, workai._x, b._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}
// OQrQ
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr256> &c, mpfr128 const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr256 workbr = type_conv<mpfr256, mpfr128>(real(b));
  mpfr256 workbi = type_conv<mpfr256, mpfr128>(imag(b));
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, a._x, workbr._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, a._x, workbi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}

// OOO
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a, std::complex<mpfr256> const &b)
{
  mpfr256 ar = real(a);
  mpfr256 ai = imag(a);
  mpfr256 br = real(b);
  mpfr256 bi = imag(b);
  mpfr256 cr1(0);
  mpfr256 ci1(0);
  mpfr_fmms(cr1._x, ar._x, br._x, ai._x, bi._x, mpfrrnd);
  mpfr256 cr = real(c);
  mpfr_add(cr._x, cr._x, cr1._x, mpfrrnd);
  mpfr_fmma(ci1._x, ar._x, bi._x, ai._x, br._x, mpfrrnd);
  mpfr256 ci = imag(c);
  mpfr_add(ci._x, ci._x, ci1._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OOOr
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a, mpfr256 const &b)
{
  mpfr256 ar = real(a);
  mpfr256 ai = imag(a);
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, ar._x, b._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, ai._x, b._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr128>(cr, ci);
}
// OOrO
template<>
void mixedp_madd<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr256> >(std::complex<mpfr256> &c, mpfr256 const &a, std::complex<mpfr256> const &b)
{
  mpfr256 br = real(b);
  mpfr256 bi = imag(b);
  mpfr256 cr = real(c);
  mpfr256 ci = imag(c);
  mpfr_fma(cr._x, a._x, br._x, cr._x, mpfrrnd);
  mpfr_fma(ci._x, a._x, bi._x, ci._x, mpfrrnd);  
  c = std::complex<mpfr256>(cr, ci);
}
#endif
} // namespace tmblas
