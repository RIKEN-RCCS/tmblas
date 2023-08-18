//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TRSV_HPP
#define _TRSV_HPP

namespace tmblas {

// =============================================================================
/// Solve the triangular matrix-vector equation
/// \[
///     op(A) x = b,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// x and b are vectors,
/// and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// @see LAPACK's latrs for a more numerically robust implementation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero.
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///
/// @param[in] trans
///     The equation to be solved:
///     - Op::NoTrans:   $A   x = b$,
///     - Op::Trans:     $A^T x = b$,
///     - Op::ConjTrans: $A^H x = b$.
///
/// @param[in] diag
///     Whether A has a unit or non-unit diagonal:
///     - Diag::Unit:    A is assumed to be unit triangular.
///                      The diagonal elements of A are not referenced.
///     - Diag::NonUnit: A is not assumed to be unit triangular.
///
/// @param[in] n
///     Number of rows and columns of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @param[in, out] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param w
///     Work array of size n.
///     Will be allocated/deallocated within the function if not given.
///
/// @ingroup trsv

template<typename Ta, typename Tb, typename Td = blas::scalar_type<Ta, Tb> > 
void
trsv(blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n, Ta const *A, const idx_int lda, Tb *X,
	  const idx_int incx,
          Td *w = nullptr);


#ifdef CBLAS_ROUTINES
template<>
void
trsv<float, float>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  float const *A, const idx_int lda,
	  float *X, const idx_int incx,
	  float *w);

// all complex float
template<>
void
trsv<std::complex<float>, std::complex<float> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<float> const *A, const idx_int lda,
	  std::complex<float> *X,
	  const idx_int incx,
	  std::complex<float> *w);

// all real double
template<>
void
trsv<double, double>(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  double const *A, const idx_int lda,
	  double *X, const idx_int incx,
	  double *w);

// all complex double
template<>
void
trsv<std::complex<double>, std::complex<double> >(
          blas::Uplo uplo,
	  blas::Op transA,
	  blas::Diag diag,
	  const idx_int n,
	  std::complex<double> const *A, const idx_int lda,
	  std::complex<double> *X,
	  const idx_int incx,
	  std::complex<double> *w);
#endif

}

#endif
