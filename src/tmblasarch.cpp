//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// #define DD_REAL
#include "tmblasarch.hpp"

//template<typename T>
//std::string tostring(T const &x) { std::string dummy; return dummy; };

namespace tmblas {
  template<typename T>
  T sqrt1( T &x ) {
    return sqrt(x);
  }

  template
  half sqrt1( half &x );
  
  template
  float sqrt1( float &x );

  template
  double sqrt1( double &x );

  template
  quadruple sqrt1( quadruple &x );
  
  template
  octuple sqrt1( octuple &x );

  template<>
  half abs1(half const &x) {
    return fabs(x);
  }
  
  template<>
  half abs1(std::complex<half> const &x) {
    return fabs(x.real()) + fabs(x.imag());
  }

  template<>
  float abs1(float const &x) {
    return fabs(x);
  }

  template<>
  float abs1(std::complex<float> const &x) {
    return fabs(x.real()) + fabs(x.imag());
  }

  template<>
  double abs1(double const &x) {
    return fabs(x);
  }

  template<>
  double abs1(std::complex<double> const &x) {
    return fabs(x.real()) + fabs(x.imag());
  }

#ifdef QD_LIBRARY  
  template<>
  dd_real abs1(dd_real const &x) {
    return fabs(x);
  }

  template<>
  dd_real abs1(std::complex<dd_real> const &x) {
    return fabs(real<dd_real>(x)) + fabs(imag<dd_real>(x));
  }
  
  template<>
  qd_real abs1( qd_real const &x ) {
      return fabs( x );
  }
  
  template<>
  qd_real abs1( std::complex<qd_real> const &x ) {
    return fabs(real<qd_real>(x)) + fabs(imag<qd_real>(x));
  }
#elif defined(MPFR)

  template<>
  mpfr128 abs1(mpfr128 const &x) {
    mpfr128 y(x);
    if (mpfr_signbit(x._x)) {
      mpfr_neg(y._x, y._x, mpfrrnd);
      return y;
    }
    else {
      return x;
    }
  } 


  template<>
  mpfr256 abs1(mpfr256 const &x) {
    mpfr256 y(x);
    if (mpfr_signbit(x._x)) {
      mpfr_neg(y._x, y._x, mpfrrnd);
      return y;
    }
    else {
      return x;
    }
  }

  template
  mpfr128 abs1(std::complex<mpfr128> const &x);
  
  template
  mpfr256 abs1(std::complex<mpfr256> const &x);
  
#elif defined(GNU_FLOAT128)
  template<>
  octuple abs1( octuple const &x )
  {
    return octuple(nullptr);
  }
  
  template<>
  octuple abs1( std::complex<octuple> const &x )
  {
    return octuple(nullptr);    
  }
#endif

  
#ifdef HALF
template<>
std::string tostring<half>(half const &x)  {
  char buf[256];
  sprintf(buf, "%6.4e", double(x));
  return std::string(buf);
}

template<>
std::string tostring<std::complex<half> >(std::complex<half> const &x)  {
  char buf[256];
  sprintf(buf, "(%6.4e %6.4e)", double(x.real()), double(x.imag()));
  return std::string(buf);
}
#endif

template<>
std::string tostring<float>(float const &x)  {
  char buf[256];
  sprintf(buf, "%12.8e", x);
  return std::string(buf);
}
template<>
std::string tostring<std::complex<float> >(std::complex<float> const &x)  {
  char buf[256];
  sprintf(buf, "(%12.8e %12.8e)", x.real(), x.imag());
  return std::string(buf);
}

template<>
std::string tostring<double>(double const &x)  {
  char buf[256];
  sprintf(buf, "%24.16e", x);
  return std::string(buf);
}
template<>
std::string tostring<std::complex<double> >(std::complex<double> const &x)  {
  char buf[256];
  sprintf(buf, "(%24.16e %24.16e)", x.real(), x.imag());
  return std::string(buf);
}

#ifdef QD_LIBRARY
template<>
std::string tostring<dd_real>(dd_real const &x)  {
#if 0
  char buf[256];
  sprintf(buf, "%24.16e", to_double(x));
  return std::string(buf);
#else
    return x.to_string();
#endif
  //  return x.to_string(dd_real::_ndigits);
}

template<>
std::string tostring<std::complex<dd_real> >(std::complex<dd_real> const &x) {
#if 0
  char buf[256];
  sprintf(buf, "(%24.16e %24.16e)", to_double(x.real()), to_double(x.imag()));
  return std::string(buf);
#else
  return "( " + real<dd_real>(x).to_string(dd_real::_ndigits) + " "
              + imag<dd_real>(x).to_string(dd_real::_ndigits) + " )";
#endif
}

template<>
std::string tostring<qd_real>(qd_real const &x)  {
#if 0
  char buf[256];
  sprintf(buf, "%24.16e", to_double(x));
  return std::string(buf);
#else
  return x.to_string(qd_real::_ndigits);
#endif
}

template<>
std::string tostring<std::complex<qd_real> >(std::complex<qd_real> const &x) {
#if 0
  return std::string("foo");
  char buf[256];
  sprintf(buf, "(%24.16e %24.16e)", to_double(x.real()), to_double(x.imag()));
#else
  return "( " + real<qd_real>(x).to_string(qd_real::_ndigits) + " "
              + imag<qd_real>(x).to_string(qd_real::_ndigits) + " )";
#endif
}
#elif defined(MPFR)
  template<>
std::string tostring<mpfr128>(mpfr128 const &x)  {
  mpfr::real_exp_t exp;  
  std::string tmp(mpfr_get_str(0, &exp, 10, 0, x._x, MPFR_RNDN));
  std::string result;
    char buf[256];
    sprintf(buf, "e%+03d", (int)exp - 1);  
  if (tmp[0] == '-') {
    result = tmp.substr(0,2)+"." + tmp.substr(2) + std::string(buf);
  }
  else {
    result = tmp.substr(0,1)+"." + tmp.substr(1) + std::string(buf);
  }
  return result;
}

template<>
std::string tostring<mpfr256>(mpfr256 const &x)  {
  mpfr::real_exp_t exp;  
  std::string tmp(mpfr_get_str(0, &exp, 10, 0,  x._x, MPFR_RNDN));
  std::string result;
  char buf[256];
  sprintf(buf, "e%+03d", (int)exp - 1);
  if (tmp[0] == '-') {
    result = tmp.substr(0,2)+"." + tmp.substr(2) + std::string(buf);
  }
  else {
    result = tmp.substr(0,1)+"." + tmp.substr(1) + std::string(buf);
  }
  return result;
}
  
template<>
std::string tostring<std::complex<mpfr128> >(std::complex<mpfr128> const &x)  {
  mpfr::real_exp_t exp;  
  std::string tmpr(mpfr_get_str(0, &exp, 10, 0, real<mpfr128>(x)._x, MPFR_RNDN));
  std::string resultr, resulti;
  char buf[256];
  sprintf(buf, "e%+03d", (int)exp - 1);  

  if (tmpr[0] == '-') {
    resultr = tmpr.substr(0,2)+"." + tmpr.substr(2) + std::string(buf);
  }
  else {
    resultr = tmpr.substr(0,1)+"." + tmpr.substr(1) + std::string(buf);
  }
  std::string tmpi(mpfr_get_str(0, &exp, 10, 0, imag<mpfr128>(x)._x, MPFR_RNDN));  
  sprintf(buf, "e%+03d", (int)exp - 1);    
  if (tmpi[0] == '-') {
    resulti = tmpi.substr(0,2)+"." + tmpi.substr(2) + std::string(buf);
  }
  else {
    resulti = tmpi.substr(0,1)+"." + tmpi.substr(1) + std::string(buf);
  }
  
  return "( " + resultr + " " + resulti + " )";
}

template<>
std::string tostring<std::complex<mpfr256> >(std::complex<mpfr256> const &x)  {
  mpfr::real_exp_t exp;  
  std::string tmpr(mpfr_get_str(0, &exp, 10, 0, real<mpfr256>(x)._x, MPFR_RNDN));
  std::string resultr, resulti;
  char buf[256];
  sprintf(buf, "e%+03d", (int)exp - 1);

  if (tmpr[0] == '-') {
    resultr = tmpr.substr(0,2)+"." + tmpr.substr(2) + std::string(buf);
  }
  else {
    resultr = tmpr.substr(0,1)+"." + tmpr.substr(1) + std::string(buf);
  }
  std::string tmpi(mpfr_get_str(0, &exp, 10, 0, imag<mpfr256>(x)._x, MPFR_RNDN));  
  sprintf(buf, "e%+03d", (int)exp - 1);    
  
  if (tmpi[0] == '-') {
    resulti = tmpi.substr(0,2)+"." + tmpi.substr(2) + std::string(buf);
  }
  else {
    resulti = tmpi.substr(0,1)+"." + tmpi.substr(1) + std::string(buf);
  }
  
  return "( " + resultr + " " + resulti + " )";
}

  
#else
#ifdef CLANG_LONG_DOUBLE
template<>
std::string tostring<quadruple>(quadruple const &x)  {
  char buf[256];
  sprintf(buf, "%40.32Le", x);
  return std::string(buf);
}

template<>
std::string tostring<std::complex<quadruple> >(std::complex<quadruple> const &x) {
  char buf[256];
  sprintf(buf, "( %40.32Le %40.32Le )", x.real(), x.imag());
  return std::string(buf);
}
#else
template<>
std::string tostring<quadruple>(quadruple const &x)  {
  char buf[256];
  quadmath_snprintf(buf, 256, "%40.32Qf", x);
  return std::string(buf);
}

template<>
std::string tostring<std::complex<quadruple> >(std::complex<quadruple> const &x) {
  char buf[256];
  quadmath_snprintf(buf, 256, "( %40.32Qf %40.32Qf )", x.real(), x.imag());
  return std::string(buf);
}
#endif
#endif



template<>
double fromstring(std::string const &s)
{
  return atof(s.c_str());
}
#ifdef QD_LIBRARY
template<>  
dd_real fromstring(std::string const &s)
{
  dd_real tmp(s.c_str());
  return tmp;
}

template<>  
qd_real fromstring(std::string const &s)
{
  qd_real tmp(s.c_str());
  return tmp;

}
#elif defined(MPFR)
template<>  
mpfr128 fromstring(std::string const &s)
{
  mpfr128 tmp;
  mpfr_set_str(tmp._x, s.c_str(), 10, MPFR_RNDN);
  return tmp;
}

template<>  
mpfr256 fromstring(std::string const &s)
{
  mpfr256 tmp;
  mpfr_set_str(tmp._x, s.c_str(), 10, MPFR_RNDN);
  return tmp;
}

#endif
  

template
std::complex<half> conjg(const std::complex<half> &a);

template<>
std::complex<float> conjg(const std::complex<float> &a) {
  return std::conj(a);
}

template<>
std::complex<double> conjg(const std::complex<double> &a) {
  return std::conj(a);
}

template
std::complex<quadruple> conjg(const std::complex<quadruple> &a);

template
std::complex<octuple> conjg(const std::complex<octuple> &a);

}

//
