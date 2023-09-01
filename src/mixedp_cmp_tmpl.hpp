//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_CMP_TMPL_HPP
#define _MIXEDP_CMP_TMPL_HPP

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

}

#endif
