//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_MUL_TMPL_HPP
#define _MIXEDP_MUL_TMPL_HPP

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
  
} // namespace tmblas
#endif
