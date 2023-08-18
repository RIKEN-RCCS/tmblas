//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_TYPE_CONV_HPP
#define _MIXEDP_TYPE_CONV_HPP

namespace tmblas {

// real-real with conversion by constructor
template<typename Ta, typename Tb,
         typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
				  std::is_same<Tb, blas::real_type<Tb> >::value &&
				  !std::is_same<Ta, Tb>::value),
				 std::nullptr_t>::type = nullptr>  
blas::real_type<Ta> type_conv(blas::real_type<Tb> const &b);

// real-real without conversion
template<typename Ta, typename Tb,
   typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			    std::is_same<Tb, blas::real_type<Tb> >::value &&
			    std::is_same<Ta, Tb>::value),
			   std::nullptr_t>::type = nullptr>
blas::real_type<Tb> type_conv(blas::real_type<Tb> const &b);

// complex-complex with conversion by constructor
template<typename Ta, typename Tb,
   typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			    std::is_same<Tb, blas::complex_type<Tb> >::value &&
			    !std::is_same<Ta, Tb>::value),
			   std::nullptr_t>::type = nullptr>
blas::complex_type<Ta> type_conv(blas::complex_type<Tb> const &b);

// complex-complex without conversion

template<typename Ta, typename Tb,
    typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			     std::is_same<Tb, blas::complex_type<Tb> >::value &&
			     std::is_same<Ta, Tb>::value),
			    std::nullptr_t>::type = nullptr>
blas::complex_type<Tb> type_conv(blas::complex_type<Tb> const &b);

// complex-real with conversion in real function

template<typename Ta, typename Tb,
    typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			   std::is_same<Tb, blas::real_type<Tb> >::value &&
	     !std::is_same<blas::real_type<Ta>, blas::real_type<Tb> >::value),
			    std::nullptr_t>::type = nullptr>  
blas::complex_type<Ta> type_conv(blas::real_type<Tb> const &b);
// complex-real without conversion

template<typename Ta, typename Tb,
  typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			   std::is_same<Tb, blas::real_type<Tb> >::value &&
	   std::is_same<blas::real_type<Ta>, blas::real_type<Tb> >::value),
			  std::nullptr_t>::type = nullptr>  
blas::complex_type<Tb> type_conv(blas::real_type<Tb> const &b);
// explicit instantiation for type_conv,
// for no convesion without reinterpret_cast
// real 
  
#ifdef QD_LIBRARY
// DQ
template<>
double type_conv<double, dd_real>(dd_real const &b);
// DO
template<>
double type_conv<double, qd_real>(qd_real const &b);
// QO
template<>
dd_real type_conv<dd_real, qd_real>(qd_real const &b);

#elif defined(MPFR)
// DQ downcast
template<>
double type_conv<double, mpfr128>(mpfr128 const &b);
// DO downcast
template<>
double type_conv<double, mpfr256>(mpfr256 const &b);
// QD upcast
template<>
mpfr128 type_conv<mpfr128, double>(double const &b);
// OD upcast
template<>
mpfr256 type_conv<mpfr256, double>(double const &b);
// QO downcast
template<>
mpfr128 type_conv<mpfr128, mpfr256>(mpfr256 const &b);
// OQ upcast
template<>
mpfr256 type_conv<mpfr256, mpfr128>(mpfr128 const &b);
#endif
  
} // namespace

#endif

