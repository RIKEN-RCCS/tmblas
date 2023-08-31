//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_MSUB_TMPL_HPP
#define _MIXEDP_MSUB_TMPL_HPP

#include "tmblasarch.hpp"

namespace tmblas {
    // Td == Ta == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				    std::nullptr_t>::type = nullptr >
  void mixedp_msub(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Td> Tdreal;    
    msub<Tdreal>(c, a, b);
  }
  // Td != Ta && Td == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(!std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >
  void mixedp_msub(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    msub<Tdreal>(c, worka, b);
  }
  // Td == Ta && Td != Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    !std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >  
  void mixedp_msub(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    msub<Tdreal>(c, a, workb);
  }
  // Td != Ta && Td != Tb
template<typename Td, typename Ta, typename Tb,
   typename std::enable_if<!(std::is_same<Td, Ta>::value ||
			      std::is_same<Td, Tb>::value),
			    std::nullptr_t>::type = nullptr >
  void mixedp_msub(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    msub<Tdreal>(c, worka, workb);
  }

  template<typename Td, typename Ta, typename Tb>
   void mixedp_msub(blas::complex_type<Td> &c,
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
    
    Tdreal cr(real<Tdreal>(c)), ci(imag<Tdreal>(c));
    // cr -= ar * br - ai * bi;
    // ci -= ar * bi + ai * br;    
    mixedp_msub<Tdreal, Tareal, Tbreal>(cr, ar, br);
    mixedp_madd<Tdreal, Tareal, Tbreal>(cr, ai, bi);
    mixedp_msub<Tdreal, Tareal, Tbreal>(ci, ar, bi);
    mixedp_msub<Tdreal, Tareal, Tbreal>(ci, ai, br);    
    c = std::complex<Tdreal>(cr, ci);
  }

  template<typename Td, typename Ta, typename Tb>
  void mixedp_msub(blas::complex_type<Td> &c,
		   blas::complex_type<Ta> const &a,
		   blas::real_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr(real<Tdreal>(c)), ci(imag<Tdreal>(c));
     mixedp_msub<Tdreal, Tareal, Tbreal>(cr, real<Tareal>(a), b);
     mixedp_msub<Tdreal, Tareal, Tbreal>(ci, imag<Tareal>(a), b);     
     c = std::complex<Tdreal>(cr, ci);
  }
  template<typename Td, typename Ta, typename Tb>
  void mixedp_msub(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::complex_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr(real<Tdreal>(c)), ci(imag<Tdreal>(c));
     mixedp_msub<Tdreal, Tareal, Tbreal>(cr, a, real<Tbreal>(b));
     mixedp_msub<Tdreal, Tareal, Tbreal>(ci, a, imag<Tbreal>(b));     
     c = std::complex<Tdreal>(cr, ci);
  }

  template<typename Td, typename Ta, typename Tb>
  void mixedp_msub(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Td> Tdreal;
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    Tdreal cr(real<Tdreal>(c));
    mixedp_msub<Tdreal, Tareal, Tbreal>(cr, a, b);
    c = std::complex<Tdreal>(cr, imag<Tdreal>(c));
  }
} // namespace tmblas
#endif
