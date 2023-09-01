//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//#define FLOATRANGE 32768.0 // 2^15 : dgemm with n = 1000
#define FLOATRANGE 1024.0 // 2^10 : zgemm with n = 1000

template<typename RC, typename TC>
void check_diff(int64_t m, int64_t n, RC *Cr, TC *Ct,
		int64_t ldc, double *errormax, double *errorl2,
		blas::Uplo uplo = blas::Uplo::General)
{
  typedef blas::scalar_type<RC, TC> scalar_t;
  typedef blas::real_type<scalar_t> realscalar_t;
  scalar_t tmp;
  *errormax = 0.0;
  *errorl2 = 0.0;
  switch(uplo) {
  case blas::Uplo::General:
    for (int64_t j = 0; j < n; j++) {    
      for (int64_t i = 0; i < m; i++) {
	tmblas::mixedp_sub<scalar_t, RC, TC>(tmp, Cr[i + j * ldc], Ct[i + j * ldc]);
	double dtmpr, dtmpi;
	dtmpr = fabs(tmblas::type_conv<double, realscalar_t >(tmblas::real<realscalar_t>(tmp)));
	if (*errormax < dtmpr) {
	  *errormax = dtmpr;
	}
	*errorl2 += dtmpr * dtmpr;
	
	if (blas::is_complex<scalar_t>::value) {
	  dtmpi = fabs(tmblas::type_conv<double, realscalar_t >(tmblas::imag<realscalar_t>(tmp)));
	  if (*errormax < dtmpi) {
	    *errormax = dtmpi;
	  }
	  *errorl2 += dtmpi * dtmpi;
	}
      }
    }
    *errorl2 = sqrt(*errorl2);
    break;
  case blas::Uplo::Upper:
    assert( m == n);
    for (int64_t j = 0; j < n; j++) {    
      for (int64_t i = 0; i <=j ; i++) {
	tmblas::mixedp_sub<scalar_t, RC, TC>(tmp, Cr[i + j * ldc], Ct[i + j * ldc]);
	double dtmpr, dtmpi;
	dtmpr = fabs(tmblas::type_conv<double, realscalar_t >(tmblas::real<realscalar_t>(tmp)));
	if (*errormax < dtmpr) {
	  *errormax = dtmpr;
	}
	*errorl2 += dtmpr * dtmpr;
	
	if (blas::is_complex<scalar_t>::value) {
	  dtmpi = fabs(tmblas::type_conv<double, realscalar_t >(tmblas::imag<realscalar_t>(tmp)));
	  if (*errormax < dtmpi) {
	    *errormax = dtmpi;
	  }
	  *errorl2 += dtmpi * dtmpi;
	}
      }
    }
    *errorl2 = sqrt(*errorl2);
      break;
    case blas::Uplo::Lower:
    assert( m == n);
    for (int64_t j = 0; j < n; j++) {    
      for (int64_t i = j; i < n; i++) {
	tmblas::mixedp_sub<scalar_t, RC, TC>(tmp, Cr[i + j * ldc], Ct[i + j * ldc]);
	double dtmpr, dtmpi;
	dtmpr = fabs(tmblas::type_conv<double, realscalar_t >(tmblas::real<realscalar_t>(tmp)));
	if (*errormax < dtmpr) {
	  *errormax = dtmpr;
	}
	*errorl2 += dtmpr * dtmpr;
	
	if (blas::is_complex<scalar_t>::value) {
	  dtmpi = fabs(tmblas::type_conv<double, realscalar_t >(tmblas::imag<realscalar_t>(tmp)));
	  if (*errormax < dtmpi) {
	    *errormax = dtmpi;
	  }
	  *errorl2 += dtmpi * dtmpi;
	}
      }
    }
    *errorl2 = sqrt(*errorl2);
    break;
  }
}

template<typename RC, typename TC>
void check_diff(int64_t m, int64_t n, RC *Cr, TC *Ct,
		int64_t ldc, double *errormax, double *errorl2, blas::Uplo uplo);

template<typename T>
void tmblas_setscal(blas::real_type<T> &c, double re, double im = 0.0)
{
  typedef blas::real_type<T> Treal;
  Treal Tre(tmblas::type_conv<Treal, double>(re));
  c = Tre;
}

template<typename T>
void tmblas_setscal(blas::complex_type<T> &c, double re, double im)
{
  typedef blas::real_type<T> Treal;
  Treal Tre(tmblas::type_conv<Treal, double>(re));
  Treal Tim(tmblas::type_conv<Treal, double>(im));
  c = std::complex<Treal>(Tre, Tim);
}


template<typename T>
T tmblas_trunc(blas::real_type<T> &x, double floatrange = FLOATRANGE)
{
  typedef blas::real_type<T> Treal;
  double y;
  y = tmblas::type_conv<double, Treal>(x);
  y = floor(y * floatrange) / floatrange;
  return tmblas::type_conv<Treal, double>(y);
}

template<typename T>
T tmblas_trunc(blas::complex_type<T> &x, double floatrange = FLOATRANGE)
{
  typedef blas::real_type<T> Treal;
  double yr, yi;
  yr = tmblas::type_conv<double, Treal>(tmblas::real<Treal>(x));
  yr = floor(yr * floatrange) / floatrange;
  yi = tmblas::type_conv<double, Treal>(tmblas::imag<Treal>(x));
  yi = floor(yi * floatrange) / floatrange;
  
  return std::complex<Treal>(tmblas::type_conv<Treal, double>(yr),
			     tmblas::type_conv<Treal, double>(yi));
}

template<typename T>
void tmblas_larnv( int64_t idist, int iseed[4], int64_t size,
		   std::complex<T> *x, bool trunc = false,
		   double floatrange = FLOATRANGE)
{
  fprintf(stderr, "%s %d tmnblas_larnv is not instantiated\n",
	  __FILE__, __LINE__);
}  

template<typename T>
void tmblas_larnv( int64_t idist, int iseed[4], int64_t size,
		   T *x, bool trunc = false,
		   double floatrange = FLOATRANGE)		   
{
  fprintf(stderr, "%s %d tmnblas_larnv is not instantiated\n",
	  __FILE__, __LINE__);
}  

template <>
void tmblas_larnv<float>( int64_t idist, int iseed[4], int64_t size,
			  float *x, bool trunc, double floatrange);


template <>
void tmblas_larnv<double>( int64_t idist, int iseed[4], int64_t size,
			   double *x, bool trunc, double floatrange);

template <>
void tmblas_larnv<quadruple>( int64_t idist, int iseed[4], int64_t size,
			      quadruple *x, bool trunc, double floatrange);

template <>
void tmblas_larnv<octuple>( int64_t idist, int iseed[4], int64_t size,
			      octuple *x, bool trunc, double floatrange);

template <>
void tmblas_larnv<float>( int64_t idist, int iseed[4], int64_t size,
			  std::complex<float> *x, bool trunc,
			  double floatrange);

template <>
void tmblas_larnv<double>( int64_t idist, int iseed[4], int64_t size,
			   std::complex<double> *x, bool trunc,
			   double floatrange);

template <>
void tmblas_larnv<quadruple>( int64_t idist, int iseed[4], int64_t size,
			      std::complex<quadruple> *x, bool truc,
			      double floatrange);			      

template <>
void tmblas_larnv<octuple>( int64_t idist, int iseed[4], int64_t size,
			    std::complex<octuple> *x, bool truc,
			    double floatrange);


// -----------------------------------------------------------------------------
typedef int64_t blas_int;
void lapack_potrf(  char uplo, int64_t n,
                    float *A, int64_t lda,
                    int64_t *info );

void lapack_potrf(  char uplo, int64_t n,
                    double *A, int64_t lda,
                    int64_t *info );

void lapack_potrf(  char uplo, int64_t n,
                    std::complex<float> *A, int64_t lda,
                    int64_t *info );

void lapack_potrf(  char uplo, int64_t n,
                    std::complex<double> *A, int64_t lda,
                    int64_t *info );

