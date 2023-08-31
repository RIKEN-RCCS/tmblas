//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _MIXEDP_TYPE_CONV_TMPL_HPP
#define _MIXEDP_TYPE_CONV_TMPL_HPP

#include "tmblasarch.hpp"

namespace tmblas {
//  type_conv

// real-real with conversion by constructor
template<typename Ta, typename Tb,
   typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			    std::is_same<Tb, blas::real_type<Tb> >::value &&
			    !std::is_same<Ta, Tb>::value),
			   std::nullptr_t>::type>  
			   //std::nullptr_t>::type = nullptr>  
blas::real_type<Ta> type_conv(blas::real_type<Tb> const &b)
{
  return Ta(b);
}

// real-real without conversion
template<typename Ta, typename Tb,
   typename std::enable_if<(std::is_same<Ta, blas::real_type<Ta> >::value &&
			    std::is_same<Tb, blas::real_type<Tb> >::value &&
			    std::is_same<Ta, Tb>::value),
			   std::nullptr_t>::type>
			   //std::nullptr_t>::type = nullptr>
blas::real_type<Tb> type_conv(blas::real_type<Tb> const &b)
{
  return b;
}

// complex-complex with conversion by constructor
template<typename Ta, typename Tb,
   typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			    std::is_same<Tb, blas::complex_type<Tb> >::value &&
			    !std::is_same<Ta, Tb>::value),
			   std::nullptr_t>::type>
			   //std::nullptr_t>::type = nullptr>
blas::complex_type<Ta> type_conv(blas::complex_type<Tb> const &b)  
{
  typedef blas::real_type<Ta> realTa;
  typedef blas::real_type<Tb> realTb;
  realTa br(type_conv<realTa, realTb>(real<realTb>(b)));
  realTa bi(type_conv<realTa, realTb>(imag<realTb>(b)));  
  return std::complex<blas::real_type<Ta> >(br, bi);
}

// complex-complex without conversion
template<typename Ta, typename Tb,
    typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			     std::is_same<Tb, blas::complex_type<Tb> >::value &&
			     std::is_same<Ta, Tb>::value),
			    std::nullptr_t>::type>
			    //std::nullptr_t>::type = nullptr>
blas::complex_type<Tb> type_conv(blas::complex_type<Tb> const &b)
{
  return b;
}
  
// complex-real with conversion in real function
template<typename Ta, typename Tb,
    typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			     std::is_same<Tb, blas::real_type<Tb> >::value &&
	     !std::is_same<blas::real_type<Ta>, blas::real_type<Tb> >::value),
			    std::nullptr_t>::type>  
			    //std::nullptr_t>::type = nullptr>  
blas::complex_type<Ta> type_conv(blas::real_type<Tb> const &b)
{
  typedef blas::real_type<Ta> realTa;
  typedef blas::real_type<Tb> realTb;  
  return std::complex<blas::real_type<Ta> >(type_conv<realTa, realTb>(b), realTa(0));
}

// complex-real without conversion
template<typename Ta, typename Tb,
  typename std::enable_if<(std::is_same<Ta, blas::complex_type<Ta> >::value &&
			   std::is_same<Tb, blas::real_type<Tb> >::value &&
	   std::is_same<blas::real_type<Ta>, blas::real_type<Tb> >::value),
			  std::nullptr_t>::type>  
			  //std::nullptr_t>::type = nullptr>  
blas::complex_type<Tb> type_conv(blas::real_type<Tb> const &b)
{
  typedef blas::real_type<Tb> realTb;  
  return std::complex<blas::real_type<Ta> >(b, realTb(0));
}

} // namespace
#endif
