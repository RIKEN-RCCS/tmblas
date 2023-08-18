//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _CSRGEMM_HPP
# define _CSRGEMM_HPP

namespace tmblas{

template<typename Ta, typename Tb, typename Tc, typename Td = blas::scalar_type<Ta, Tb, Tc> >
void csrgemm(
     blas::Op trans,
     idx_int m,
     idx_int n,     
     blas::scalar_type<Ta, Tb, Tc> const &alpha,     
     Ta const *av,
     idx_int const *ai,
     idx_int const *aj,
     Tb const *x,
     idx_int ldx,     
     blas::scalar_type<Ta, Tb, Tc> const &beta,
     Tc *y,
     idx_int ldy,
     Td *w = nullptr);
} // namespace tmblas
#endif

