//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_CMP_CPP
#define _MIXEDP_CMP_CPP

#include "tmblasarch.hpp"

//template<typename Td, typename Ta, typename Tb> 
//bool mixedp_add(Td &c, Ta const &a, Tb const &b) { // c = a + b
namespace tmblas {
// generic routine

 template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value),
		 std::nullptr_t>::type> 
		 //std::nullptr_t>::type = nullptr> 
  bool mixedp_eq(Ta const &a, Tb const &b)	 
  {
    typedef blas::real_type<Ta> Tareal;
    Tareal workb(b);     
    return ((real<Tareal>(a) == workb) &&
	    (imag<Tareal>(a) == Tareal(0)));
  }
  
  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			  std::is_same<Tb, blas::complex_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type> 
		 //std::nullptr_t>::type = nullptr> 
  bool mixedp_eq(Ta const &a, Ta const &b)	 
  {
    typedef blas::real_type<Ta> Tareal;
    return ((real<Tareal>(a) == real<Tareal>(b)) &&
	    (imag<Tareal>(a) == imag<Tareal>(b)));
    
    return (a == b);
  }
  
  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			  std::is_same<Tb, blas::complex_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type> 
		 //std::nullptr_t>::type = nullptr> 
  bool mixedp_eq(blas::complex_type<Ta> const &a, blas::complex_type<Tb> const &b)
  {
    typedef blas::complex_type<blas::scalar_type<Ta, Tb> > Tc;
    typedef blas::real_type<blas::scalar_type<Ta, Tb> > Tcreal;
    
    Tc worka = type_conv<Tc, Ta>(a);
    Tc workb = type_conv<Tc, Tb> >(b); 
    return ((real<Tcreal>(worka) == real<Tcreal>(workb)) &&
	    (imag<Tcreal>(worka) == imag<Tcreal>(workb)));
  }

  template<typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type>
		 //std::nullptr_t>::type = nullptr>
  bool mixedp_eq(Ta const &a, Ta const &b)
  {
    return (a == b);
  }

  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type>  
		 //std::nullptr_t>::type = nullptr>  
  bool mixedp_eq(Ta const &a, Tb const &b)		 
  {
    typedef blas::scalar_type<Ta, Tb> Tc;
    Tc worka = type_conv<Tc, Ta>(a);
    Tc workb = type_conv<Tc, Tb>(b);
    return (worka == workb);
  }

  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type>  
		 //std::nullptr_t>::type = nullptr>  
  bool mixedp_lt(Ta const &a, Ta const &b)
  {
    return (a < b);
  }

 template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type>  
		 //std::nullptr_t>::type = nullptr>  
  bool mixedp_lt(Ta const &a, Tb const &b)
  {
    typedef blas::scalar_type<Ta, Tb> Tc;
    Tc worka = type_conv<Tc, Ta>(a);
    Tc workb = type_conv<Tc, Tb>(b);
    return (worka < workb);
  }

 template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type>  
		 //std::nullptr_t>::type = nullptr>  
  bool mixedp_gt(Ta const &a, Ta const &b)
  {
    return (a > b);
  }
  
 template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type>  
		 //std::nullptr_t>::type = nullptr>  
  bool mixedp_gt(Ta const &a, Tb const &b)
  {
    typedef blas::scalar_type<Ta, Tb> Tc;
    Tc worka = type_conv<Tc, Ta>(a);
    Tc workb = type_conv<Tc, Tb>(b);
    return (worka > workb);
  }

// instantiations  
// real
// HH
template
bool mixedp_eq<half, half>(half const &a, half const &b);

// HF
template
bool mixedp_eq<half, float>(half const &a, float const &b);
// FH
template
bool mixedp_eq<float, half>(float const &a, half const &b);
  
//FF  
template
bool mixedp_eq<float, float>(float const &a, float const &b);

// FD
template
bool mixedp_eq<float, double>(float const &a, double const &b);
// DF
template
bool mixedp_eq<double, float>(double const &a, float const &b);

//DD  
template
bool mixedp_eq<double, double>(double const &a, double const &b);

//QQ  
template
bool mixedp_eq<quadruple, quadruple>(quadruple const &a, quadruple const &b);

//OO  
template
bool mixedp_eq<octuple, octuple>(octuple const &a, octuple const &b);

// HH
template
bool mixedp_eq<std::complex<half>, std::complex<half> >(std::complex<half> const &a, std::complex<half> const &b);

// FF
template
bool mixedp_eq<std::complex<float>, std::complex<float> >(std::complex<float> const &a, std::complex<float> const &b);

// DD
template
bool mixedp_eq<std::complex<double>, std::complex<double> >(std::complex<double> const &a, std::complex<double> const &b);

// QQ
template
bool mixedp_eq<std::complex<quadruple>, std::complex<quadruple> >(std::complex<quadruple> const &a, std::complex<quadruple> const &b);

// OO
template
  bool mixedp_eq<std::complex<octuple>, std::complex<octuple> >(std::complex<octuple> const &a, std::complex<octuple> const &b);

  // comparison with integer
template<>
bool mixedp_eq<half, int>(half const &a, int const &b)
{
  half workb(b);
  return (a == workb);
}

template<>
bool mixedp_eq<std::complex<half>, int>(std::complex<half> const &a, int const &b)
{
  half workb(b);
  half zero(0);  
  return ((real<half>(a) == workb) &&
	  (imag<half>(a) == zero));
}

template<>
bool mixedp_eq<float, int>(float const &a, int const &b)
{
  half workb(b);
  return (a == workb);
}

template<>
bool mixedp_eq<std::complex<float>, int>(std::complex<float> const &a, int const &b)
{
  float workb(b);
  float zero(0);  
  return ((real<float>(a) == workb) &&
	  (imag<float>(a) == zero));
}

  template<>
bool mixedp_eq<double, int>(double const &a, int const &b)
{
  double workb(b);
  return (a == workb);
}

template<>
bool mixedp_eq<std::complex<double>, int>(std::complex<double> const &a, int const &b)
{
  double workb(b);
  double zero(0);  
  return ((real<double>(a) == workb) &&
	  (imag<double>(a) == zero));
}

template<>
bool mixedp_eq<quadruple, int>(quadruple const &a, int const &b)
{
  quadruple workb(b);
  return (a == workb);
}

template<>
bool mixedp_eq<std::complex<quadruple>, int>(std::complex<quadruple> const &a, int const &b)
{
  quadruple workb(b);
  quadruple zero(0);  
  return ((real<quadruple>(a) == workb) &&
	  (imag<quadruple>(a) == zero));
}

template<>
bool mixedp_eq<octuple, int>(octuple const &a, int const &b)
{
  octuple workb(b);
  return (a == workb);
}

template<>
bool mixedp_eq<std::complex<octuple>, int>(std::complex<octuple> const &a, int const &b)
{
  octuple workb(b);
  octuple zero(0);  
  return ((real<octuple>(a) == workb) &&
	  (imag<octuple>(a) == zero));
}
  
// HH
template
bool mixedp_lt<half, half>(half const &a, half const &b);

//FF  
template
bool mixedp_lt<float, float>(float const &a, float const &b);

//DD  
template
bool mixedp_lt<double, double>(double const &a, double const &b);

//QQ  
template
bool mixedp_lt<quadruple, quadruple>(quadruple const &a, quadruple const &b);

//OO  
template
bool mixedp_lt<octuple, octuple>(octuple const &a, octuple const &b);

// HH
template
bool mixedp_gt<half, half>(half const &a, half const &b);

//FF  
template
bool mixedp_gt<float, float>(float const &a, float const &b);

//DD  
template
bool mixedp_gt<double, double>(double const &a, double const &b);

//QQ  
template
bool mixedp_gt<quadruple, quadruple>(quadruple const &a, quadruple const &b);

//OO  
template
bool mixedp_gt<octuple, octuple>(octuple const &a, octuple const &b);
  
//
}
#endif
