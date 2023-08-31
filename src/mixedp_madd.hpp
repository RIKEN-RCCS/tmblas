//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_MADD_HPP
#define _MIXEDP_MADD_HPP

#include "tmblasarch.hpp"

namespace tmblas {
  // generic routine
  // Td == Ta == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				    std::nullptr_t>::type = nullptr >
  void mixedp_madd(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		   blas::real_type<Td> const &b);
  // Td != Ta && Td == Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(!std::is_same<Td, Ta>::value &&
				    std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >
  void mixedp_madd(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		   blas::real_type<Td> const &b);
  // Td == Ta && Td != Tb
  template<typename Td, typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Td, Ta>::value &&
				    !std::is_same<Td, Tb>::value),
				   std::nullptr_t>::type = nullptr >  
  void mixedp_madd(blas::real_type<Td> &c, blas::real_type<Td> const &a,
		   blas::real_type<Tb> const &b);
  // Td != Ta && Td != Tb
  template<typename Td, typename Ta, typename Tb,
   typename std::enable_if<!(std::is_same<Td, Ta>::value ||
			      std::is_same<Td, Tb>::value),
			    std::nullptr_t>::type = nullptr >
  void mixedp_madd(blas::real_type<Td> &c, blas::real_type<Ta> const &a,
		   blas::real_type<Tb> const &b);
  
  template<typename Td, typename Ta, typename Tb>
  void mixedp_madd(blas::complex_type<Td> &c,
		   blas::complex_type<Ta> const &a,
		   blas::complex_type<Tb> const &b);
  
  template<typename Td, typename Ta, typename Tb>
  void mixedp_madd(blas::complex_type<Td> &c,
		   blas::complex_type<Ta> const &a,
		   blas::real_type<Tb> const &b);
  
  template<typename Td, typename Ta, typename Tb>
  void mixedp_madd(blas::complex_type<Td> &c,
		   blas::real_type<Ta> const &a,
		   blas::complex_type<Tb> const &b);
  
  template<typename Td, typename Ta, typename Tb>
  void mixedp_madd(blas::complex_type<Td> &c,
		   blas::real_type<Ta> const &a,
		   blas::real_type<Tb> const &b);
  
#if ( defined(QD_LIBRARY) || defined(MPFR) )  
// QDD
template<>
void mixedp_madd<quadruple, double, double>(quadruple &c, double const &a, double const &b);
// QQD
template<>
void mixedp_madd<quadruple, quadruple, double>(quadruple &c, quadruple const &a, double const &b);
// QQD
template<>
void mixedp_madd<quadruple, double, quadruple>(quadruple &c, double const &a, quadruple const &b);
// OQQ
template<>
void mixedp_madd<octuple, quadruple, quadruple>(octuple &c, quadruple const &a, quadruple const &b);
// OOQ
template<>
void mixedp_madd<octuple, octuple, quadruple>(octuple &c, octuple const &a, quadruple const &b);
// OQO
template<>
void mixedp_madd<octuple, quadruple, octuple>(octuple &c, quadruple const &a, octuple const &b);
#endif

//
} // namespace tmblas
#endif
