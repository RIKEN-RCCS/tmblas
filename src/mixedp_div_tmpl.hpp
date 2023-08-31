//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_DIV_TMPL_HPP
#define _MIXEDP_DIV_TMPL_HPP

#include "tmblasarch.hpp"

namespace tmblas {
  // Td == Ta == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				    std::nullptr_t>::type = nullptr >
  void mixedp_div(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Td> Tdreal;    
    div<Tdreal>(c, a, b);
  }
  // Td != Ta && Td == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(!std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >
  void mixedp_div(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    div<Tdreal>(c, worka, b);
  }
  // Td == Ta && Td != Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    !std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >  
  void mixedp_div(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    div<Tdreal>(c, a, workb);
  }
  // Td != Ta && Td != Tb
template<typename Td, typename Ta, typename Tb,
   typename std::enable_if<!(std::is_same<Td, Ta>::value ||
			      std::is_same<Td, Tb>::value),
			    std::nullptr_t>::type = nullptr >
  void mixedp_div(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    div<Tdreal>(c, worka, workb);
  }

  template<typename Td, typename Ta, typename Tb>
   void mixedp_div(blas::complex_type<Td> &c,
		   blas::complex_type<Ta> const &a,
		   blas::complex_type<Tb> const &b)
  {
    typedef blas::real_type<Td> Tdreal;
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    
    Tareal ar(real<Tareal>(a));
    Tareal ai(imag<Tareal>(a));
    Tbreal br(real<Tbreal>(b));
    Tbreal bi(imag<Tbreal>(b));	      
    Tdreal rr, cr, ci;
    mixedp_mul<Tdreal, Tbreal, Tbreal>(rr, br, br);
    mixedp_madd<Tdreal, Tbreal, Tbreal>(rr, bi, bi);  // rr = br * br + bi * bi
    mixedp_mul<Tdreal, Tareal, Tbreal>(cr, ar, br);
    mixedp_madd<Tdreal, Tareal, Tbreal>(cr, ai, bi);  
    mixedp_div<Tdreal, Tdreal, Tdreal>(cr, cr, rr);   // cr = (ar * br + ai * bi) / rr;
    mixedp_mul<Tdreal, Tareal, Tbreal>(ci, ai, br);
    mixedp_msub<Tdreal, Tareal, Tbreal>(ci, ar, bi);  
    mixedp_div<Tdreal, Tdreal, Tdreal>(ci, ci, rr);   // cr = (ai * br - ar * bi) / rr;
    c = std::complex<Tdreal>(cr, ci);
  }

  template<typename Td, typename Ta, typename Tb>
  void mixedp_div(blas::complex_type<Td> &c,
		  blas::complex_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr, ci;
     mixedp_div<Tdreal, Tareal, Tbreal>(cr, real<Tareal>(a), b);
     mixedp_div<Tdreal, Tareal, Tbreal>(ci, imag<Tareal>(a), b);     
     c = std::complex<Tdreal>(cr, ci);
  }
  template<typename Td, typename Ta, typename Tb>
  void mixedp_div(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::complex_type<Tb> const &b)
  {
    typedef blas::real_type<Td> Tdreal;
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    
    Tbreal br(real<Tbreal>(b));
    Tbreal bi(imag<Tbreal>(b));	      
    Tdreal rr, cr, ci;
    mixedp_mul<Tdreal, Tbreal, Tbreal>(rr, br, br);
    mixedp_madd<Tdreal, Tbreal ,Tbreal>(rr, bi, bi);  // rr = br * br + bi * bi
    mixedp_mul<Tdreal, Tareal, Tbreal>(cr, a, br);
    mixedp_div<Tdreal, Tdreal, Tdreal>(cr, cr, rr);   // cr = a * br / rr;
    mixedp_mul<Tdreal, Tareal, Tbreal>(ci, a, bi);
    mixedp_div<Tdreal, Tdreal, Tdreal>(ci, ci, rr);   // cr = - a * bi / rr;
    c = std::complex<Tdreal>(cr, -ci);
  }

  template<typename Td, typename Ta, typename Tb>
  void mixedp_div(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Td> Tdreal;
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    Tdreal cr;
    mixedp_div<Tdreal, Tareal, Tbreal>(cr, a, b);
    c = std::complex<Tdreal>(cr, Tdreal(0));
  }
  
} // namespace tmblas
#endif
