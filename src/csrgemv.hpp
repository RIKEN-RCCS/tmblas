//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRGEMV_HPP
# define _CSRGEMV_HPP

namespace tmblas{

template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> >
void csrgemv(
     blas::Op trans,
     idx_int m,
     blas::scalar_type<Ta, Tb, Tc> const &alpha,     
     Ta const *av,
     idx_int const *ai,
     idx_int const *aj,
     Tb const *x,
     blas::scalar_type<Ta, Tb, Tc> const &beta,
     Tc *y,
     Td *w = nullptr);
} // namespace tmblas
#endif

