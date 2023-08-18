//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_CMP_HPP
#define _MIXEDP_CMP_HPP

namespace tmblas {
// generic routine

 template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value),
		 std::nullptr_t>::type = nullptr> 
 bool mixedp_eq(Ta const &a, Tb const &b);
  
  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			  std::is_same<Tb, blas::complex_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr> 
  bool mixedp_eq(Ta const &a, Ta const &b);
  
  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			  std::is_same<Tb, blas::complex_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr> 
  bool mixedp_eq(blas::complex_type<Ta> const &a, blas::complex_type<Tb> const &b);

  template<typename Ta, typename Tb,
	   typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr>
  bool mixedp_eq(Ta const &a, Ta const &b);

  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr>  
  bool mixedp_eq(Ta const &a, Tb const &b);

  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr> 
  bool mixedp_lt(Ta const &a, Ta const &b);

  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr>  
    bool mixedp_lt(Ta const &a, Tb const &b);

  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr>  
    bool mixedp_gt(Ta const &a, Ta const &b);
  
  template<typename Ta, typename Tb,
 typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			  std::is_same<Tb, blas::real_type<Tb> >::value &&
			  !std::is_same<Ta, Tb>::value),
		 std::nullptr_t>::type = nullptr>  
    bool mixedp_gt(Ta const &a, Tb const &b);

  // comparison with integer
template<>
bool mixedp_eq<half, int>(half const &a, int const &b);

template<>
bool mixedp_eq<std::complex<half>, int>(std::complex<half> const &a, int const &b);

template<>
bool mixedp_eq<float, int>(float const &a, int const &b);

template<>
bool mixedp_eq<std::complex<float>, int>(std::complex<float> const &a, int const &b);

template<>
bool mixedp_eq<double, int>(double const &a, int const &b);


template<>
bool mixedp_eq<std::complex<double>, int>(std::complex<double> const &a, int const &b);

template<>
bool mixedp_eq<quadruple, int>(quadruple const &a, int const &b);

template<>
bool mixedp_eq<std::complex<quadruple>, int>(std::complex<quadruple> const &a, int const &b);

template<>
bool mixedp_eq<octuple, int>(octuple const &a, int const &b);

template<>
bool mixedp_eq<std::complex<octuple>, int>(std::complex<octuple> const &a, int const &b);
  
//
}
#endif
