//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_MUL_CPP
#define _MIXEDP_MUL_CPP

#include "tmblasarch.hpp"

namespace tmblas {
  // real
  // Td == Ta == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				    std::nullptr_t>::type = nullptr >
  void mixedp_mul(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Td> Tdreal;    
    mul<Tdreal>(c, a, b);
  }
  // Td != Ta && Td == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(!std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >
  void mixedp_mul(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    mul<Tdreal>(c, worka, b);
  }
  // Td == Ta && Td != Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    !std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >  
  void mixedp_mul(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    mul<Tdreal>(c, a, workb);
  }
  // Td != Ta && Td != Tb
template<typename Td, typename Ta, typename Tb,
   typename std::enable_if<!(std::is_same<Td, Ta>::value ||
			      std::is_same<Td, Tb>::value),
			    std::nullptr_t>::type = nullptr >
  void mixedp_mul(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    mul<Tdreal>(c, worka, workb);
  }

  template<typename Td, typename Ta, typename Tb>
   void mixedp_mul(blas::complex_type<Td> &c,
		   blas::complex_type<Ta> const &a,
		   blas::complex_type<Tb> const &b)
  {
    typedef blas::real_type<Td> Tdreal;
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;    
    Tareal ar = real<Tareal>(a);
    Tareal ai = imag<Tareal>(a);
    Tbreal br = real<Tbreal>(b);
    Tbreal bi = imag<Tbreal>(b);
    
    Tdreal cr, ci;
    // cr = ar * br - ai * bi;
    // ci = ar * bi + ai * br;    
    mixedp_mul<Tdreal, Tareal, Tbreal>(cr, ar, br);
    mixedp_msub<Tdreal, Tareal, Tbreal>(cr, ai, bi);
    mixedp_mul<Tdreal, Tareal, Tbreal>(ci, ar, bi);
    mixedp_madd<Tdreal, Tareal, Tbreal>(ci, ai, br);    
    c = std::complex<Tdreal>(cr, ci);
  }

  template<typename Td, typename Ta, typename Tb>
  void mixedp_mul(blas::complex_type<Td> &c,
		  blas::complex_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr, ci;
     mixedp_mul<Tdreal, Tareal, Tbreal>(cr, real<Tareal>(a), b);
     mixedp_mul<Tdreal, Tareal, Tbreal>(ci, imag<Tareal>(a), b);     
     c = std::complex<Tdreal>(cr, ci);
  }
  template<typename Td, typename Ta, typename Tb>
  void mixedp_mul(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::complex_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr, ci;
     mixedp_mul<Tdreal, Tareal, Tbreal>(cr, a, real<Tbreal>(b));
     mixedp_mul<Tdreal, Tareal, Tbreal>(ci, a, imag<Tbreal>(b));     
     c = std::complex<Tdreal>(cr, ci);
  }

  template<typename Td, typename Ta, typename Tb>
  void mixedp_mul(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Td> Tdreal;
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    Tdreal cr;
    mixedp_mul<Tdreal, Tareal, Tbreal>(cr, a, b);
    c = std::complex<Tdreal>(cr, Tdreal(0));
  }
  
// instantiations
// real
// HHH
template
void mixedp_mul<half, half, half>(half &c, half const &a, half const &b);
// FHH
template
void mixedp_mul<float, half, half>(float &c, half const &a, half const &b);
// FFH
template
void mixedp_mul<float, float, half>(float &c, float const &a, half const &b);
// FHF
template
void mixedp_mul<float, half, float>(float &c, half const &a, float const &b);
// FFF
template
void mixedp_mul<float, float, float>(float &c, float const &a, float const &b);
// DFF
template
void mixedp_mul<double, float, float>(double &c, float const &a, float const &b);
// DDF
template
void mixedp_mul<double, double, float>(double &c, double const &a, float const &b);
template
// DFD
void mixedp_mul<double, float, double>(double &c, float const &a, double const &b);
// DDD
template
void mixedp_mul<double, double, double>(double &c, double const &a, double const &b);
//

#ifdef QD_LIBRARY
// QDD
template<>
void mixedp_mul<quadruple, double, double>(quadruple &c, double const &a, double const &b)
{
  c = type_conv<quadruple, double>(a) * b; // dedicated function c = a * b with two double?
  //  std::cout << __FILE__ << " " << __LINE__ << std::endl;  
}

// QQD
template<>
void mixedp_mul<quadruple, quadruple, double>(quadruple &c, quadruple const &a, double const &b)
{
  c = a * b; // dedicated function c = a * b with two double?
  //  std::cout << __FILE__ << " " << __LINE__ << std::endl;  
}

// QDQ
template<>
void mixedp_mul<quadruple, double, quadruple>(quadruple &c, double const &a, quadruple const &b)
{
  c = a * b;
  //    std::cout << __FILE__ << " " << __LINE__ << std::endl;
}

// QQQ
template<>
void mixedp_mul<quadruple, quadruple, quadruple>(quadruple &c, quadruple const &a, quadruple const &b)
{
  mul<quadruple>(c, a, b);  // mono-precision
  //    std::cout << __FILE__ << " " << __LINE__ << std::endl;
}

// OQQ
template<>
void mixedp_mul<octuple, quadruple, quadruple>(octuple &c, quadruple const &a, quadruple const &b)
{
  c = type_conv<octuple, quadruple>(a) * b; // dedicated function c = a * b with two quadruple?
  //    std::cout << __FILE__ << " " << __LINE__ << std::endl;
}

// OOQ
template<>
void mixedp_mul<octuple, octuple, quadruple>(octuple &c, octuple const &a, quadruple const &b)
{
  c = a * b; // dedicated function c = a * b with two quadruple?
  //  std::cout << __FILE__ << " " << __LINE__ << std::endl;  
}

// OQO
template<>
void mixedp_mul<octuple, quadruple, octuple>(octuple &c, quadruple const &a, octuple const &b)
{
  c = a * b;
  //    std::cout << __FILE__ << " " << __LINE__ << std::endl;
}

// OOO
template<>
void mixedp_mul<octuple, octuple, octuple>(octuple &c, octuple const &a, octuple const &b)
{
  mul<octuple>(c, a, b);  // mono-precision
  //  std::cout << __FILE__ << " " << __LINE__ << std::endl;
}

#elif defined(MPFR)
// QDD
template<>
void mixedp_mul<mpfr128, double, double>(mpfr128 &c, double const &a, double const &b) {
  mpfr128 d;
  mpfr_set_d(d._x, a, mpfrrnd);  // upcasting a
  mpfr_mul_d(c._x, d._x, b, mpfrrnd);
}

// QQD
template<>
void mixedp_mul<mpfr128, mpfr128, double>(mpfr128 &c, mpfr128 const &a, double const &b) {
  mpfr_mul_d(c._x, a._x, b, mpfrrnd);
}

// QDQ
template<>
void mixedp_mul<mpfr128, double, mpfr128>(mpfr128 &c, double const &a, mpfr128 const &b) {
  mpfr_mul_d(c._x, b._x, a, mpfrrnd); // no mpfr_d_mul()
}

// QQQ
template<>
void mixedp_mul<mpfr128, mpfr128, mpfr128 >(mpfr128 &c, mpfr128 const &a, mpfr128 const &b) {
  mpfr_mul(c._x, a._x, b._x, mpfrrnd);
}

// OQO
template<>
void mixedp_mul<mpfr256, mpfr128, mpfr256 >(mpfr256 &c, mpfr128 const &a,
		 mpfr256 const &b) {
  mpfr_mul(c._x, a._x, b._x, mpfrrnd);
}

// OOQ
template<>
void mixedp_mul<mpfr256, mpfr256, mpfr128>(mpfr256 &c, mpfr256 const &a,
		 mpfr128 const &b) {
  mpfr_mul(c._x, a._x, b._x, mpfrrnd);
}

// OQQ
template<>
void mixedp_mul<mpfr256, mpfr128, mpfr128>(mpfr256 &c, mpfr128 const &a,
		 mpfr128 const &b) {
  mpfr256 worka = type_conv<mpfr128, mpfr128>(a);  
  mpfr_mul(c._x, worka._x, b._x, mpfrrnd);
}

// OOO
template<>
void mixedp_mul<mpfr256, mpfr256, mpfr256 >(mpfr256 &c, mpfr256 const &a, mpfr256 const &b) {
  mpfr_mul(c._x, a._x, b._x, mpfrrnd);
}

#else
// QDD
template
void mixedp_mul<quadruple, double, double>(quadruple &c, double const &a, double const &b);
// QQD
template
void mixedp_mul<quadruple, quadruple, double>(quadruple &c, quadruple const &a, double const &b);
// QDQ
template
void mixedp_mul<quadruple, double, quadruple>(quadruple &c, double const &a, quadruple const &b);
// QQQ
template<>
void mixedp_mul<quadruple, quadruple, quadruple>(quadruple &c, quadruple const &a, quadruple const &b);
//OQO
template
void mixedp_mul<octuple, quadruple, octuple>(octuple &c, quadruple const &a, octuple const &b);
// OOQ
template
void mixedp_mul<octuple, octuple, quadruple>(octcuple &c, octuple const &a, quadruple const &b);
// OQQ
template
void mixedp_mul<octuple, quadruple, quadruple>(octuple &c, quadruple const &a, quadruple const &b);
// OOO
template
void mixedp_mul<octuple, octuple, octuple>(octuple &c, octuple const &a, octuple const &b);
//
#endif


// complex
// HHH
template
void mixedp_mul<std::complex<half>, std::complex<half>, std::complex<half> >(std::complex<half> &c, std::complex<half> const &a, std::complex<half> const &b);
// HHHr
template
void mixedp_mul<std::complex<half>, std::complex<half>, half>(std::complex<half> &c, std::complex<half> const &a, half const &b);
// HHrH
template
void mixedp_mul<std::complex<half>, half, std::complex<half> >(std::complex<half> &c, half const &a, std::complex<half> const &b);
// HHrHr
template
void mixedp_mul<std::complex<half>, half, half>(std::complex<half> &c, half const &a, half const &b);

// FHH
template
void mixedp_mul<std::complex<float>, std::complex<half>, std::complex<half> >(std::complex<float> &c, std::complex<half> const &a, std::complex<half> const &b);
// FHHr
template
void mixedp_mul<std::complex<float>, std::complex<half>, half>(std::complex<float> &c, std::complex<half> const &a, half const &b);
// FHrH
template
void mixedp_mul<std::complex<float>, half, std::complex<half> >(std::complex<float> &c, half const &a, std::complex<half> const &b);
// FHrHr
template
void mixedp_mul<std::complex<float>, half, half>(std::complex<float> &c, half const &a, half const &b);
  
// FFH
template
void mixedp_mul<std::complex<float>, std::complex<float>, std::complex<half> >(std::complex<float> &c, std::complex<float> const &a, std::complex<half> const &b);
// FFHr
template
void mixedp_mul<std::complex<float>, std::complex<float>, half>(std::complex<float> &c, std::complex<float> const &a, half const &b);
// FFrH
template
void mixedp_mul<std::complex<float>, float, std::complex<half> >(std::complex<float> &c, float const &a, std::complex<half> const &b);
// FFrHr
template
void mixedp_mul<std::complex<float>, float, half>(std::complex<float> &c, float const &a, half const &b);


//FHF
template
void mixedp_mul<std::complex<float>, std::complex<half>, std::complex<float> >(std::complex<float> &c, std::complex<half> const &a, std::complex<float> const &b);
//FHFr
template
void mixedp_mul<std::complex<float>, std::complex<half>, float>(std::complex<float> &c, std::complex<half> const &a, float const &b);
//FHrF
template
void mixedp_mul<std::complex<float>, half, std::complex<float> >(std::complex<float> &c, half const &a, std::complex<float> const &b);
//FHrFr
template
void mixedp_mul<std::complex<float>, half, float>(std::complex<float> &c, half const &a, float const &b);

 //FFF
template
void mixedp_mul<std::complex<float>, std::complex<float>, std::complex<float> >(std::complex<float> &c, std::complex<float> const &a, std::complex<float> const &b);

//FFFr
template
void mixedp_mul<std::complex<float>, std::complex<float>, float>(std::complex<float> &c, std::complex<float> const &a, float const &b);

//FFrF
template
void mixedp_mul<std::complex<float>, float, std::complex<float> >(std::complex<float> &c, float const &a, std::complex<float> const &b);

//FFrFr
template
void mixedp_mul<std::complex<float>, float, float>(std::complex<float> &c, float const &a, float const &b);

//DFF
template
void mixedp_mul<std::complex<double>, std::complex<float>, std::complex<float> >(std::complex<double> &c, std::complex<float> const &a, std::complex<float> const &b);
//DFFr
template
void mixedp_mul<std::complex<double>, std::complex<float>, float>(std::complex<double> &c, std::complex<float> const &a, float const &b);
//DFrF
template
void mixedp_mul<std::complex<double>, float, std::complex<float> >(std::complex<double> &c, float const &a, std::complex<float> const &b);
//DFrFr
template
void mixedp_mul<std::complex<double>, float, float>(std::complex<double> &c, float const &a, float const &b);

// DDF
template
void mixedp_mul<std::complex<double>, std::complex<double>, std::complex<float> >(std::complex<double> &c, std::complex<double> const &a, std::complex<float> const &b);
// DDFr
template
void mixedp_mul<std::complex<double>, std::complex<double>, float>(std::complex<double> &c, std::complex<double> const &a, float const &b);
// DDrF
template
void mixedp_mul<std::complex<double>, double, std::complex<float> >(std::complex<double> &c, double const &a, std::complex<float> const &b);
// DDrFr
template
void mixedp_mul<std::complex<double>, double, float>(std::complex<double> &c, double const &a, float const &b);

// DFD
template
void mixedp_mul<std::complex<double>, std::complex<float>, std::complex<double> >(std::complex<double> &c, std::complex<float> const &a, std::complex<double> const &b);
// DFDr
template
void mixedp_mul<std::complex<double>, std::complex<float>, double>(std::complex<double> &c, std::complex<float> const &a, double const &b);
// DFrD
template
void mixedp_mul<std::complex<double>, float, std::complex<double> >(std::complex<double> &c, float const &a, std::complex<double> const &b);
// DFrDr
template
void mixedp_mul<std::complex<double>, float, double>(std::complex<double> &c, float const &a, double const &b);

//DDD
template
void mixedp_mul<std::complex<double>, std::complex<double>, std::complex<double> >(std::complex<double> &c, std::complex<double> const &a, std::complex<double> const &b);
//DDDr
template
void mixedp_mul<std::complex<double>, std::complex<double>, double>(std::complex<double> &c, std::complex<double> const &a, double const &b);
//DDrD
template
void mixedp_mul<std::complex<double>, double, std::complex<double> >(std::complex<double> &c, double const &a, std::complex<double> const &b);
//DDrDr
template
void mixedp_mul<std::complex<double>, double, double>(std::complex<double> &c, double const &a, double const &b);

// QDD
template
void mixedp_mul<std::complex<quadruple>, std::complex<double>, std::complex<double> >(std::complex<quadruple> &c, std::complex<double> const &a, std::complex<double> const &b);

// QDDr
template
void mixedp_mul<std::complex<quadruple>, std::complex<double>, double>(std::complex<quadruple> &c, std::complex<double> const &a, double const &b);

// QDrD
template
void mixedp_mul<std::complex<quadruple>, double, std::complex<double> >(std::complex<quadruple> &c, double const &a, std::complex<double> const &b);

// QDrDr
template
void mixedp_mul<std::complex<quadruple>, double, double>(std::complex<quadruple> &c, double const &a, double const &b);

// QQD
template
void mixedp_mul<std::complex<quadruple>, std::complex<quadruple>, std::complex<double> >(std::complex<quadruple> &c, std::complex<quadruple> const &a, std::complex<double> const &b);

// QQDr
template
void mixedp_mul<std::complex<quadruple>, std::complex<quadruple>, double>(std::complex<quadruple> &c, std::complex<quadruple> const &a, double const &b);

// QQrD
template
void mixedp_mul<std::complex<quadruple>, quadruple, std::complex<double> > (std::complex<quadruple> &c, quadruple const &a, std::complex<double> const &b);

// QQrDr
template
void mixedp_mul<std::complex<quadruple>, quadruple, double> (std::complex<quadruple> &c, quadruple const &a, double const &b);

// QDQ
template
void mixedp_mul<std::complex<quadruple>, std::complex<double>, std::complex<quadruple> >(std::complex<quadruple> &c, std::complex<double> const &a, std::complex<quadruple> const &b);

// QDQr
template
void mixedp_mul<std::complex<quadruple>, std::complex<double>, quadruple>(std::complex<quadruple> &c, std::complex<double> const &a, quadruple const &b);

// QDrQ
template
void mixedp_mul<std::complex<quadruple>, double, std::complex<quadruple> >(std::complex<quadruple> &c, double const &a, std::complex<quadruple> const &b);

// QDrQr
template
void mixedp_mul<std::complex<quadruple>, double, quadruple>(std::complex<quadruple> &c, double const &a, quadruple const &b);

// QQQ
template
void mixedp_mul<std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >(std::complex<quadruple> &c, std::complex<quadruple> const &a, std::complex<quadruple> const &b);

// QQQr
template
void mixedp_mul<std::complex<quadruple>, std::complex<quadruple>, quadruple>(std::complex<quadruple> &c, std::complex<quadruple> const &a, quadruple const &b);

// QQrQ
template
void mixedp_mul<std::complex<quadruple>, quadruple, std::complex<quadruple> >(std::complex<quadruple> &c, quadruple const &a, std::complex<quadruple> const &b);

// QQrQr
template
void mixedp_mul<std::complex<quadruple>, quadruple, quadruple >(std::complex<quadruple> &c, quadruple const &a, quadruple const &b);

// OQQ
template
void mixedp_mul<std::complex<octuple>, std::complex<quadruple>, std::complex<quadruple> >(std::complex<octuple> &c, std::complex<quadruple> const &a, std::complex<quadruple> const &b);
// OQQr
template
void mixedp_mul<std::complex<octuple>, std::complex<quadruple>, quadruple>(std::complex<octuple> &c, std::complex<quadruple> const &a, quadruple const &b);
// OQrQ
template
void mixedp_mul<std::complex<octuple>, quadruple, std::complex<quadruple> >(std::complex<octuple> &c, quadruple const &a, std::complex<quadruple> const &b);
// OQrQr
template
void mixedp_mul<std::complex<octuple>, quadruple, quadruple>(std::complex<octuple> &c, quadruple const &a, quadruple const &b);

// OOQ
template
void mixedp_mul<std::complex<octuple>, std::complex<octuple>, std::complex<quadruple> >(std::complex<octuple> &c, std::complex<octuple> const &a, std::complex<quadruple> const &b);

// OOQr
template
void mixedp_mul<std::complex<octuple>, std::complex<octuple>, quadruple>(std::complex<octuple> &c, std::complex<octuple> const &a, quadruple const &b);

// OOrQ
template
void mixedp_mul<std::complex<octuple>, octuple, std::complex<quadruple> >(std::complex<octuple> &c, octuple const &a, std::complex<quadruple> const &b);

// OQO
template
void mixedp_mul<std::complex<octuple>, std::complex<quadruple>, std::complex<octuple> >(std::complex<octuple> &c, std::complex<quadruple> const &a, std::complex<octuple> const &b);
// OQOr
template
void mixedp_mul<std::complex<octuple>, std::complex<quadruple>, octuple>(std::complex<octuple> &c, std::complex<quadruple> const &a, octuple const &b);
// OQrO
template
void mixedp_mul<std::complex<octuple>, quadruple, std::complex<octuple> >(std::complex<octuple> &c, quadruple const &a, std::complex<octuple> const &b);
// OQrOr
template
void mixedp_mul<std::complex<octuple>, quadruple, octuple>(std::complex<octuple> &c, quadruple const &a, octuple const &b);

// OOO
template
void mixedp_mul<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(std::complex<octuple> &c, std::complex<octuple> const &a, std::complex<octuple> const &b);
// OOOr
template
void mixedp_mul<std::complex<octuple>, std::complex<octuple>, octuple>(std::complex<octuple> &c, std::complex<octuple> const &a, octuple const &b);
// OOrO
template
void mixedp_mul<std::complex<octuple>, octuple, std::complex<octuple> >(std::complex<octuple> &c, octuple const &a, std::complex<octuple> const &b);
// OOrOr
template
void mixedp_mul<std::complex<octuple>, octuple, octuple>(std::complex<octuple> &c, octuple const &a, octuple const &b);

#if 0 // fmms/fmma in MPFR for future improvements
// QDD
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<double>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<double> const &a, std::complex<double> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));
  mpfr128 workbr = type_conv<mpfr128, double>(real(b));
  mpfr128 workbi = type_conv<mpfr128, double>(imag(b));

  mpfr_fmms(cr._x, workar._x, workbr._x, workai._x, workbi._x, mpfrrnd);
  mpfr_fmma(ci._x, workar._x, workbi._x, workai._x, workbr._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QDDr
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<double>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<double> const &a, double const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));

  mpfr_mul_d(cr._x, workar._x, b, mpfrrnd);
  mpfr_mul_d(ci._x, workai._x, b, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QDrD
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<double>, std::complex<double> >(std::complex<mpfr128> &c, double const &a, std::complex<double> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr128 workbr = type_conv<mpfr128, double>(real(b));
  mpfr128 workbi = type_conv<mpfr128, double>(imag(b));

  mpfr_mul_d(cr._x, workbr._x, a, mpfrrnd);
  mpfr_mul_d(ci._x, workbi._x, a, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}

// QQD
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, std::complex<double> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr128 workbr = type_conv<mpfr128, double>(real(b));
  mpfr128 workbi = type_conv<mpfr128, double>(imag(b));
  mpfr_fmms(cr._x, real(a)._x, workbr._x, imag(a)._x, workbi._x, mpfrrnd);
  mpfr_fmma(ci._x, real(a)._x, workbi._x, imag(a)._x, workbr._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}

// QQDr
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<double> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, double const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr_mul_d(cr._x, real(a)._x, b, mpfrrnd);
  mpfr_mul_d(ci._x, imag(a)._x, b, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}

// QQrD
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<double> >(std::complex<mpfr128> &c, mpfr128 const &a, std::complex<double> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr_mul_d(cr._x, a._x, real(b), mpfrrnd);
  mpfr_mul_d(ci._x, a._x, imag(b), mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}

// QDQ
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<double>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<double> const &a, std::complex<mpfr128> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));
  mpfr_fmms(cr._x, workar._x, real(b)._x, workai._x, imag(b)._x, mpfrrnd);
  mpfr_fmma(ci._x, workar._x, imag(b)._x, workai._x, real(b)._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QDQr
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<double>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<double> const &a, mpfr128 const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr128 workar = type_conv<mpfr128, double>(real(a));
  mpfr128 workai = type_conv<mpfr128, double>(imag(a));
  mpfr_mul_d(cr._x, b._x, real(a), mpfrrnd);
  mpfr_mul_d(ci._x, b._x, imag(a), mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}

// QDrQ
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<double>, std::complex<mpfr128> >(std::complex<mpfr128> &c, double const &a, std::complex<mpfr128> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr_mul_d(cr._x, real(b)._x, a, mpfrrnd);
  mpfr_mul_d(ci._x, imag(b)._x, a, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}


// QQQ
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, std::complex<mpfr128> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr_fmms(cr._x, real(a)._x, real(b)._x, imag(a)._x, imag(b)._x, mpfrrnd);
  mpfr_fmma(ci._x, real(a)._x, imag(b)._x, imag(a)._x, real(b)._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QQQr
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr128> &c, std::complex<mpfr128> const &a, mpfr128 const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(a)._x, b._x, mpfrrnd);
  mpfr_mul(ci._x, imag(a)._x, b._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}
// QQrQ
template<>
void mixedp_mul<std::complex<mpfr128>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr128> &c, mpfr128 const &a, std::complex<mpfr128> const &b)
{
  mpfr128 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(b)._x, a._x, mpfrrnd);
  mpfr_mul(ci._x, imag(b)._x, a._x, mpfrrnd);
  c = std::complex<mpfr128>(cr, ci);
}

// OQO
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 std::complex<mpfr256> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_fmms(cr._x, real(a)._x, real(b)._x, imag(a)._x, imag(b)._x, mpfrrnd);
  mpfr_fmma(ci._x, real(a)._x, imag(b)._x, imag(a)._x, real(b)._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OQOr
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 mpfr256 const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(a)._x, b._x, mpfrrnd);
  mpfr_mul(ci._x, imag(a)._x, b._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OQrO
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr256> >(std::complex<mpfr256> &c, mpfr128 const &a,
		 std::complex<mpfr256> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(b)._x, a._x, mpfrrnd);
  mpfr_mul(ci._x, imag(b)._x, a._x,  mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}

// OOQ
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_fmms(cr._x, real(a)._x, real(b)._x, imag(a)._x, imag(b)._x, mpfrrnd);
  mpfr_fmma(ci._x, real(a)._x, imag(b)._x, imag(a)._x, real(b)._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OOQr
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a,
		 mpfr128 const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(a)._x, b._x, mpfrrnd);
  mpfr_mul(ci._x, real(a)._x, b._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OOrQ
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr128> >(std::complex<mpfr256> &c, mpfr256 const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(b)._x, a._x, mpfrrnd);
  mpfr_mul(ci._x, imag(b)._x, a._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}

// OQQ
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_fmms(cr._x, real(a)._x, real(b)._x, imag(a)._x, imag(b)._x, mpfrrnd);
  mpfr_fmma(ci._x, real(a)._x, imag(b)._x, imag(a)._x, real(b)._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OQQr
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr256> &c, std::complex<mpfr128> const &a,
		 mpfr128 const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(a)._x, b._x, mpfrrnd);
  mpfr_mul(ci._x, imag(a)._x, b._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OQrQ
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr128>, std::complex<mpfr128> >(std::complex<mpfr256> &c, mpfr128 const &a,
		 std::complex<mpfr128> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(b)._x, a._x, mpfrrnd);
  mpfr_mul(ci._x, imag(b)._x, a._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}

// OOO
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a, std::complex<mpfr256> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_fmms(cr._x, real(a)._x, real(b)._x, imag(a)._x, imag(b)._x, mpfrrnd);
  mpfr_fmma(ci._x, real(a)._x, imag(b)._x, imag(a)._x, real(b)._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OOOr
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr256> >(std::complex<mpfr256> &c, std::complex<mpfr256> const &a, mpfr256 const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(a)._x, b._x, mpfrrnd);
  mpfr_mul(ci._x, imag(a)._x, b._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
// OOrO
template<>
void mixedp_mul<std::complex<mpfr256>, std::complex<mpfr256>, std::complex<mpfr256> >(std::complex<mpfr256> &c, mpfr256 const &a, std::complex<mpfr256> const &b)
{
  mpfr256 cr(0.0), ci(0.0);
  mpfr_mul(cr._x, real(b)._x, a._x, mpfrrnd);
  mpfr_mul(ci._x, imag(b)._x, a._x, mpfrrnd);
  c = std::complex<mpfr256>(cr, ci);
}
#endif
} // namespace tmblas
#endif
