//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_SUB_TMPL_HPP
#define _MIXEDP_SUB_TMPL_HPP

#include "tmblasarch.hpp"

namespace tmblas {
  // real
  // Td == Ta == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				    std::nullptr_t>::type>
				    //std::nullptr_t>::type = nullptr >
  void mixedp_sub(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Td> Tdreal;    
    sub<Tdreal>(c, a, b);
  }
  // Td != Ta && Td == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(!std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type>
				   //std::nullptr_t>::type = nullptr >
  void mixedp_sub(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Td> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    sub<Tdreal>(c, worka, b);
  }
  // Td == Ta && Td != Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    !std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type>  
				   //std::nullptr_t>::type = nullptr >  
  void mixedp_sub(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    sub<Tdreal>(c, a, workb);
  }
  // Td != Ta && Td != Tb
template<typename Td, typename Ta, typename Tb,
   typename std::enable_if<!(std::is_same<Td, Ta>::value ||
			      std::is_same<Td, Tb>::value),
			    std::nullptr_t>::type>
			    //std::nullptr_t>::type = nullptr >
  void mixedp_sub(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
    Tdreal worka = type_conv<Tdreal, Tareal>(a);
    Tdreal workb = type_conv<Tdreal, Tbreal>(b);
    sub<Tdreal>(c, worka, workb);
  }
 // complex following real mixedp_sub
  template<typename Td, typename Ta, typename Tb>
   void mixedp_sub(blas::complex_type<Td> &c,
		   blas::complex_type<Ta> const &a,
		   blas::complex_type<Tb> const &b)
  {
    typedef blas::real_type<Ta> Tareal;
    typedef blas::real_type<Tb> Tbreal;
    typedef blas::real_type<Td> Tdreal;
     Tdreal cr, ci;
     mixedp_sub<Tdreal, Tareal, Tbreal>(cr, real<Tareal>(a), real<Tbreal>(b));
     mixedp_sub<Tdreal, Tareal, Tbreal>(ci, imag<Tareal>(a), imag<Tbreal>(b));
     c = std::complex<Tdreal>(cr, ci);
  }
  // complex-complex-real
  template<typename Td, typename Ta, typename Tb>
   void mixedp_sub(blas::complex_type<Td> &c,
		  blas::complex_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr;
     mixedp_sub<Tdreal, Tareal, Tbreal>(cr, real<Tareal>(a), b);
     Tdreal ci(type_conv<Tdreal, Tareal>(imag<Tareal>(a)));
     c = std::complex<Tdreal>(cr, ci);
  }
  // complex-real-complex  
  template<typename Td, typename Ta, typename Tb>
  void mixedp_sub(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::complex_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr;
     mixedp_sub<Tdreal, Tareal, Tbreal>(cr, a, real<Tbreal>(b));
     Tdreal ci(type_conv<Tdreal, Tbreal>(imag<Tbreal>(b)));
     c = std::complex<Tdreal>(cr, -ci);
  }

  // complex-real-real  
  template<typename Td, typename Ta, typename Tb>
  void mixedp_sub(blas::complex_type<Td> &c,
		  blas::real_type<Ta> const &a,
		  blas::real_type<Tb> const &b)
  {
     typedef blas::real_type<Td> Tdreal;
     typedef blas::real_type<Ta> Tareal;
     typedef blas::real_type<Tb> Tbreal;
     Tdreal cr;
     mixedp_sub<Tdreal, Tareal, Tbreal>(cr, a, b);
     c = std::complex<Tdreal>(cr, Tdreal(0));
  }

} // namespace tmblas
#endif
