//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TBLASARCH_HPP
# define _TBLASARCH_HPP

#ifdef INTEL_MKL
#undef CBLAS_ROUTINES
#define CBLAS_ROUTINES
#include <mkl_cblas.h>
typedef MKL_INT BLAS_INT;
typedef void BLAS_VOID;
#else
#if defined(VECLIB) || defined(OPENBLAS) || defined(ATLAS)
#undef CBLAS_ROUTINES
#define CBLAS_ROUTINES
#include <cblas.h>
typedef int BLAS_INT;
typedef void BLAS_VOID;
#else // wihtout CBLAS header files nor routines
#if 0
typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef int BLAS_INT;
typedef void BLAS_VOID;
#endif
#endif
#endif
#ifdef QD_LIBRARY
#include "qd/qd_real.h"
#include "qd/dd_real.h"
#endif

#if (defined(HALF_HARDWARE) || defined(HALF_EMULATION))
#define HALF
#endif

typedef int idx_int;
#ifdef HALF
#ifdef HALF_HARDWARE
typedef __fp16 half; // typedef bfloat16 half;
#elif defined(HALF_EMULATION)
#include "half.hpp"
typedef half_float::half half;
#endif
#else
typedef short half; // only for test
#endif

#ifdef QD_LIBRARY
#include "qd/dd_real.h"
#include "qd/qd_real.h"
typedef dd_real quadruple; // quadruple precision by double-double
typedef qd_real octuple;   // octuple precision by quad-double
#elif defined(MPFR)
#define MPFR_REAL_DATA_PUBLIC 1
#include "real.hpp"
typedef mpfr::real<113, MPFR_RNDN> mpfr128;
typedef mpfr128 quadruple;
typedef mpfr::real<237, MPFR_RNDN> mpfr256;
typedef mpfr256 octuple;
#define mpfrrnd MPFR_RNDN 
#else
#ifdef __clang__
  #define CLANG_LONG_DOUBLE
  typedef long double quadruple;  // long double = 128bit
#elif (defined(__FLT128_MAX__) || defined (__ICC))
  #define GNU_FLOAT128
  #include <quadmath.h>
  #include <array>
typedef __float128 quadruple;     // __float128 works in ICC
typedef std::array<double, 4> dummyoct;    // dummy
typedef dummyoct octuple;    // dummy
#endif
#endif
#if defined(INTEL_MKL) || defined(OPENBLAS) || defined(VECLIB) || defined(ATLAS)
#define CBLAS_ROUTINES
#endif


#include "util.hh"


namespace blas {

  template<>
  struct real_type_traits<int>
  {
    using real_t = int;
  };

  template<>
  struct real_type_traits<std::complex<int> >
  {
    using real_t = int;
  };

  template<>
  struct real_type_traits<half>
  {
    using real_t = half;
  };

  template<>
  struct real_type_traits<std::complex<half> >
  {
    using real_t = half;
  };

  template<>
  struct scalar_type_traits< half, float >
  {
    using type = decay_t<float>;
  };

  template<>
  struct scalar_type_traits< float, half >
  {
    using type = decay_t<float>;
  };

  template<>
  struct scalar_type_traits< std::complex<half>, std::complex<float> >
  {
    using type = decay_t< std::complex<float> >;
  };

  template<>
  struct scalar_type_traits< std::complex<float>, std::complex<half> >
  {
    using type = decay_t< std::complex<float> >;
  };
  
  
  template<>
  struct scalar_type_traits< octuple, quadruple >
  {
    using type = decay_t<octuple>;
  };

  template<>
  struct scalar_type_traits< quadruple, octuple >
  {
    using type = decay_t<octuple>;
  };

  template<>
  struct scalar_type_traits< double, quadruple >
  {
    using type = decay_t<quadruple>;
  };

  template<>
  struct scalar_type_traits<quadruple, double >
  {
    using type = decay_t<quadruple>;
  };

  template<>
  struct scalar_type_traits< std::complex<octuple>, std::complex<quadruple> >
  {
    using type = decay_t< std::complex<octuple> >;
  };

  template<>
  struct scalar_type_traits< std::complex<quadruple>, std::complex<octuple> >
  {
    using type = decay_t< std::complex<octuple> >;
  };

  template<>
  struct scalar_type_traits< std::complex<double>, std::complex<quadruple> >
  {
    using type = decay_t< std::complex<quadruple> >;
  };
  template<>
  struct scalar_type_traits< std::complex<quadruple>, std::complex<double> >
  {
    using type = decay_t< std::complex<quadruple> >;
  };
}

#ifdef CBLAS_ROUTINES
namespace blas {
  inline CBLAS_TRANSPOSE op2cblas( Op op ) {
    CBLAS_TRANSPOSE t = CBLAS_TRANSPOSE::CblasNoTrans;
    if(op == Op::NoTrans)
      t = CBLAS_TRANSPOSE::CblasNoTrans;
    else if (op == Op::Trans)
      t = CBLAS_TRANSPOSE::CblasTrans;
    else if (op == Op::ConjTrans)
      t = CBLAS_TRANSPOSE::CblasConjTrans;
    return t;
  }
  
  inline CBLAS_UPLO uplo2cblas( Uplo   uplo   ) {
    return (uplo == Uplo::Upper ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower);
  }
  
  inline CBLAS_DIAG diag2cblas( Diag   diag   ) {
    return (diag == Diag::NonUnit ? CBLAS_DIAG::CblasNonUnit : CBLAS_DIAG::CblasUnit);
  }
  
  inline CBLAS_SIDE side2cblas( Side   side   ) {
    return (side == Side::Left ? CBLAS_SIDE::CblasLeft : CBLAS_SIDE::CblasRight);
  }
}
#endif

namespace tmblas {

  enum class Op     : char { NoTrans   = 'N',
                             Trans     = 'T',
			     ConjTrans = 'C',
                             Conj      = 'R'};
  

#if 0
  template<typename T> blas::real_type<T> real(blas::complex_type<T> &x) {
    return x.real();
  }
  
  template<typename T> blas::real_type<T> imag(blas::complex_type<T> &x) {
    return x.imag();
  }
#endif
  template<typename T> inline T real(std::complex<T> &x)
  {
    return x.real();
  }
  
  template<typename T> inline T imag(std::complex<T> &x)
  {
    return x.imag();
  }

  template<typename T> inline blas::real_type<T> real(blas::real_type<T> &x)
  {
    return x;
  }

  template<typename T> inline blas::real_type<T> imag(blas::real_type<T> &x)
  {
    typedef blas::real_type<T> Treal;
    return Treal(0);
  }

  template<typename T>
  inline T abs1(T const &x)
  {
    if (x > T(0)) {
      return x;
    }
    else {
      return -x;
    }
  }

  template<typename T>
  inline T abs1(std::complex<T> const &x)
  {
    typedef blas::real_type<T> Treal;
    return abs1<Treal>(x.real()) + abs1<Treal>(x.imag());
  }

  template<>
  half abs1(half const &x);

  template<>
  half abs1(std::complex<half> const &x);

  template<>
  float abs1(float const &x);

  template<>
  float abs1(std::complex<float> const &x);

  template<>
  double abs1(double const &x);

  template<>
  double abs1(std::complex<double> const &x);

#ifdef QD_LIBRARY  
  template<>
  dd_real abs1(dd_real const &x);

  template<>
  dd_real abs1(std::complex<dd_real> const &x);
  
  template<>
  qd_real abs1( qd_real const &x );
  
  template<>
  qd_real abs1( std::complex<qd_real> const &x );

#elif defined(MPFR)
  template<>
  mpfr128 abs1(mpfr128 const &x);

  template<>
  mpfr256 abs1(mpfr256 const &x);

#elif defined(GNU_FLOAT128)
  template<>
  octuple abs1( octuple const &x );
  
  template<>
  octuple abs1( std::complex<octuple> const &x );
#endif

  // addition in mono-precision : c = a + b
template<typename Td>
void add(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void add(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  Tdreal cr = ar + br;
  Tdreal ci = ai + bi;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void add(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	 blas::real_type<Td> const &b);

template<typename Td>
void add(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	 blas::real_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);

  Tdreal cr = ar + b;
  c = std::complex<Tdreal>(cr, ai);
}

template<typename Td>
void add(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	 blas::complex_type<Td> const &b);

template<typename Td>
void add(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	 blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  Tdreal cr = a + br;
  c = std::complex<Tdreal>(cr, bi);
}
  
template<typename Td>
void add(Td &c, Td const &a, Td const &b);

template<typename Td>
void add(Td &c, Td const &a, Td const &b) {
  c = a + b;
}

// subtraction mono-precision : c = a - b
template<typename Td>
void sub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void sub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  Tdreal cr = ar - br;
  Tdreal ci = ai - bi;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void sub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	 blas::real_type<Td> const &b);

template<typename Td>
void sub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::real_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);

  Tdreal cr = ar - b;
  c = std::complex<Tdreal>(cr, ai);
}

template<typename Td>
void sub(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	 blas::complex_type<Td> const &b);

template<typename Td>
void sub(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  Tdreal cr = a - br;
  c = std::complex<Tdreal>(cr, bi);
}

template<typename Td>
void sub(Td &c, Td const &a, Td const &b);

template<typename Td>
void sub(Td &c, Td const &a, Td const &b) {
  c = a - b;
}

// multiplication mono-precision : c = a * b
template<typename Td>
void mul(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void mul(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  Tdreal cr = ar * br - ai * bi;
  Tdreal ci = ar * bi + ai * br;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void mul(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	 blas::real_type<Td> const &b);

template<typename Td>
void mul(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::real_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);

  Tdreal cr = ar * b;
  Tdreal ci = ai * b;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void mul(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	 blas::complex_type<Td> const &b);

template<typename Td>
void mul(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  Tdreal cr = a * br;
  Tdreal ci = a * bi;
  c = std::complex<Tdreal>(cr, ci);
}


template<typename Td>
void mul(Td &c, Td const &a, Td const &b);

template<typename Td>
void mul(Td &c, Td const &a, Td const &b) {
  c = a * b;
}

  // multiplication mono-precision : c = a * b
template<typename Td>
void div(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void div(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);
  Tdreal br = real(b);
  Tdreal bi = imag(b);
  Tdreal rr = br * br + bi * bi;
  Tdreal cr = (ar * br + ai * bi) / rr;
  Tdreal ci = (ai * br - ar * bi) / rr;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void div(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	 blas::real_type<Td> const &b);

template<typename Td>
void div(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::real_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal ar = real(a);
  Tdreal ai = imag(a);
  Tdreal cr = ar / b;
  Tdreal ci = ai / b;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void div(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	 blas::complex_type<Td> const &b);

template<typename Td>
void div(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal br = real(b);
  Tdreal bi = imag(b);
  Tdreal rr = br * br + bi * bi;
  Tdreal cr = a * br / rr;
  Tdreal ci =  - a * bi / rr;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void div(Td &c, Td const &a, Td const &b);

template<typename Td>
void div(Td &c, Td const &a, Td const &b) {
  c = a / b;
}

  
// multiplying and addition in mono-precision : c += a * b
template<typename Td>
void madd(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void madd(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal cr = real(c);
  Tdreal ci = imag(c);
  Tdreal ar = real(a);
  Tdreal ai = imag(a);
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  cr += ar * br - ai * bi;
  ci += ar * bi + ai * br;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void madd(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::real_type<Td> const &b);

template<typename Td>
void madd(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::real_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal cr = real(c);
  Tdreal ci = imag(c);
  Tdreal ar = real(a);
  Tdreal ai = imag(a);

  cr += ar * b;
  ci += ai * b;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void madd(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void madd(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal cr = real(c);
  Tdreal ci = imag(c);
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  cr += a * br;
  ci += a * bi;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void madd(Td &c, Td const &a, Td const &b);

template<typename Td>
void madd(Td &c, Td const &a, Td const &b) {
  c += a * b;
}

// multiplying and subtraction in mono-precision : c -= a * b
template<typename Td>
void msub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void msub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal cr = real(c);
  Tdreal ci = imag(c);
  Tdreal ar = real(a);
  Tdreal ai = imag(a);
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  cr -= ar * br - ai * bi;
  ci -= ar * bi + ai * br;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void msub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::real_type<Td> const &b);

template<typename Td>
void msub(blas::complex_type<Td> &c, blas::complex_type<Td> const &a,
	  blas::real_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal cr = real(c);
  Tdreal ci = imag(c);
  Tdreal ar = real(a);
  Tdreal ai = imag(a);

  cr -= ar * b;
  ci -= ai * b;
  c = std::complex<Tdreal>(cr, ci);
}

template<typename Td>
void msub(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	  blas::complex_type<Td> const &b);

template<typename Td>
void msub(blas::complex_type<Td> &c, blas::real_type<Td> const &a,
	  blas::complex_type<Td> const &b)
{
  typedef blas::real_type<Td> Tdreal;
  Tdreal cr = real(c);
  Tdreal ci = imag(c);
  Tdreal br = real(b);
  Tdreal bi = imag(b);

  cr -= a * br;
  ci -= a * bi;
  c = std::complex<Tdreal>(cr, ci);
}

// multiplying and addition in mono-precision : c += a * b  
template<typename Td>
void msub(Td &c, Td const &a, Td const &b);

template<typename Td>
void msub(Td &c, Td const &a, Td const &b) {
  c -= a * b;
}
  
  template<typename T>
  std::string tostring(T const &x);

  template<typename T>
  T fromstring(std::string const &s);

  template<typename T>
  T sqrt1( T &x );

#if 1
  template<typename Ta>
  blas::complex_type<Ta> conjg(const blas::complex_type<Ta> &a);

  template<typename Ta>
  inline blas::complex_type<Ta> conjg(const blas::complex_type<Ta> &a)
  {
    typedef blas::real_type<Ta> Tareal;
    return std::complex<Tareal>(real<Tareal>(a), -imag<Tareal>(a));
  }
#else
  template<typename Ta>
  std::complex<Ta> conjg(const std::complex<Ta> &a);

  template<typename Ta>
  std::complex<Ta> conjg(const std::complex<Ta> &a)
  {
    return std::complex<Ta>(real<Ta>(a), -imag<Ta>(a));
  }

#endif
  template<typename Ta>
  Ta conjg(const Ta &a);

  template<typename Ta>
  Ta conjg(const Ta &a) {
     return a;
  }

  template<>
  std::complex<float> conjg(const std::complex<float> &a);

  template<>
  std::complex<double> conjg(const std::complex<double> &a);
} // namespace tmblas

#include "mixedp_type_conv.hpp"
#include "mixedp_msub.hpp"
#include "mixedp_madd.hpp"
#include "mixedp_cmp.hpp"
#include "mixedp_sub.hpp"
#include "mixedp_add.hpp"
#include "mixedp_mul.hpp"
#include "mixedp_div.hpp"

#endif
