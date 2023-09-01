//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#include "test.hh"

#include "blas/flops.hh"
#include "tmblastest.hh"

template
void check_diff(int64_t m, int64_t n, float *Cr, float *Ct,
		int64_t ldc, double *errormax, double *errorl2, blas::Uplo uplo);

template
void check_diff(int64_t m, int64_t n, double *Cr, double *Ct,
		int64_t ldc, double *errormax, double *errorl2, blas::Uplo uplo);

//template
//void check_diff(int64_t m, int64_t n, quadruple *Cr, quadruple *Ct,
//		int64_t ldc, double *errormax, double *errorl2, blas::Uplo uplo);

template
void check_diff(int64_t m, int64_t n, std::complex<float> *Cr, std::complex<float> *Ct,
		int64_t ldc, double *errormax, double *errorl2, blas::Uplo uplo);

template
void check_diff(int64_t m, int64_t n, std::complex<double> *Cr, std::complex<double> *Ct,
		int64_t ldc, double *errormax, double *errorl2, blas::Uplo uplo);

//template
//void check_diff(int64_t m, int64_t n, std::complex<quadruple> *Cr, std::complex<quadruple> *Ct,
//		int64_t ldc, double *errormax, double *errorl2, blas::Uplo uplo);

template
void tmblas_setscal<float>(float &c, double re, double im);
template
void tmblas_setscal<double>(double &c, double re, double im);
template
void tmblas_setscal<quadruple>(quadruple &c, double re, double im);
template
void tmblas_setscal<octuple>(octuple &c, double re, double im);
template
void tmblas_setscal<std::complex<float> >(std::complex<float> &c, double re, double im);
template
void tmblas_setscal<std::complex<double> >(std::complex<double> &c, double re, double im);
template
void tmblas_setscal<std::complex<quadruple> >(std::complex<quadruple> &c, double re, double im);
template
void tmblas_setscal<std::complex<octuple> >(std::complex<octuple> &c, double re, double im);

template <>
void tmblas_larnv<float>( int64_t idist, int iseed[4], int64_t size,
			  float *x, bool trunc,
			  double floatrange)
{
  static const double m_const = 4.6566128730773926e-10;  /* = 2^{-31} */
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double d;
      float xx;
      d = lrand48() * m;
      xx = 2.0 * d - 1.0;
      x[i] = float(floor(xx * floatrange) / floatrange);
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double d;
      d = lrand48() * m;
      x[i] = float(2.0 * d - 1.0);
    }
  }
}

template <>
void tmblas_larnv<double>( int64_t idist, int iseed[4], int64_t size,
			   double *x, bool trunc,
			   double floatrange )
{
  static const double m_const = 4.6566128730773926e-10;  /* = 2^{-31} */
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double d, xx;
      d = lrand48() * m;
      xx = (2.0 * d - 1.0);
      x[i] = floor(xx * floatrange) / floatrange;
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double d;
      d = lrand48() * m;
      x[i] = (2.0 * d - 1.0);
    }
  }
}

/*
template <>
void tmblas_larnv<quadruple>( int64_t idist, int iseed[4], int64_t size,
			      quadruple *x, bool trunc,
			      double floatrange )
{
  static const double m_const = 4.6566128730773926e-10;  // = 2^{-31} 
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double d, xx;
      d = lrand48() * m;
      xx = (2.0 * d - 1.0);
      x[i] = tmblas::type_conv<quadruple, double>(floor(xx * floatrange) / floatrange);
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      quadruple xx(0);
      for (int j = 0; j < 4; j++, m *= m_const) {
	double d;      
	d = lrand48() * m;
	tmblas::mixedp_add<quadruple, quadruple, double>(xx, xx, d);
      }
      x[i] = quadruple(-1);
      tmblas::mixedp_madd<quadruple, quadruple, quadruple>(x[i], xx, quadruple(2));
    }
  }
}

template <>
void tmblas_larnv<octuple>( int64_t idist, int iseed[4], int64_t size,
			    octuple *x, bool trunc,
			    double floatrange )
{
  static const double m_const = 4.6566128730773926e-10;  // = 2^{-31} 
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double d, xx;
      d = lrand48() * m;
      xx = (2.0 * d - 1.0);
      x[i] = tmblas::type_conv<octuple, double>(floor(xx * floatrange) / floatrange);
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      octuple xx(0);
      for (int j = 0; j < 4; j++, m *= m_const) {
	double d;      
	d = lrand48() * m;
	octuple od(d);
	tmblas::mixedp_add<octuple, octuple, octuple>(xx, xx, od);      
      }
      x[i] = octuple(-1);
      tmblas::mixedp_madd<octuple, octuple, octuple>(x[i], xx, octuple(2));
    }
  }
}
*/

template <>
void tmblas_larnv<float>( int64_t idist, int iseed[4], int64_t size,
			  std::complex<float> *x, bool trunc,
			  double floatrange )			  
{
  static const double m_const = 4.6566128730773926e-10;  /* = 2^{-31} */  
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      
      dr = lrand48() * m;
      di = lrand48() * m;
      dr = 2.0 * dr - 1.0;
      dr = floor(dr * floatrange) / floatrange;
      di = 2.0 * di - 1.0;      
      di = floor(di * floatrange) / floatrange;
      x[i] = std::complex<float>(float(dr), float(di));
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      dr = lrand48() * m;
      di = lrand48() * m;      
      x[i] = std::complex<float>(float(2.0 * dr - 1.0), float(2.0 * di - 1.0));
    }
  }
}
template <>
void tmblas_larnv<double>( int64_t idist, int iseed[4], int64_t size,
			   std::complex<double> *x, bool trunc,
			   double floatrange )			   
{
  static const double m_const = 4.6566128730773926e-10;  /* = 2^{-31} */
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      dr = lrand48() * m;
      di = lrand48() * m;
      dr = 2.0 * dr - 1.0;
      dr = floor(dr * floatrange) / floatrange;
      di = 2.0 * di - 1.0;      
      di = floor(di * floatrange) / floatrange;
      x[i] = std::complex<double>(dr, di);
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      dr = lrand48() * m;
      di = lrand48() * m;      
      x[i] = std::complex<double>((2.0 * dr - 1.0), (2.0 * di - 1.0));
    }
  }
}

/*
template <>
void tmblas_larnv<quadruple>( int64_t idist, int iseed[4], int64_t size,
			      std::complex<quadruple> *x, bool trunc,
			      double floatrange )
{
  static const double m_const = 4.6566128730773926e-10;  // = 2^{-31} 
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      dr = lrand48() * m;
      di = lrand48() * m;
      dr = 2.0 * dr - 1.0;
      dr = floor(dr * floatrange) / floatrange;
      di = 2.0 * di - 1.0;      
      di = floor(di * floatrange) / floatrange;
      x[i] = std::complex<quadruple>(tmblas::type_conv<quadruple, double>(dr),
				     tmblas::type_conv<quadruple, double>(di));
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      quadruple xr(0), xi(0);
      for (int j = 0; j < 4; j++, m *= m_const) {
	dr = lrand48() * m;
	di = lrand48() * m;
	tmblas::mixedp_add<quadruple, quadruple, double>(xr, xr, dr);
	tmblas::mixedp_add<quadruple, quadruple, double>(xi, xi, di);      
      }
      quadruple xxr(-1), xxi(-1);
      tmblas::mixedp_madd<quadruple, quadruple, quadruple>(xxr, xr, quadruple(2));
      tmblas::mixedp_madd<quadruple, quadruple, quadruple>(xxi, xi, quadruple(2));    
      x[i] = std::complex<quadruple>(xxr, xxi);
    }
  }
}

template <>
void tmblas_larnv<octuple>( int64_t idist, int iseed[4], int64_t size,
			    std::complex<octuple> *x, bool trunc,
			    double floatrange )
{
  static const double m_const = 4.6566128730773926e-10;  // = 2^{-31} 
  if (trunc) {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      dr = lrand48() * m;
      di = lrand48() * m;
      dr = 2.0 * dr - 1.0;
      dr = floor(dr * floatrange) / floatrange;
      di = 2.0 * di - 1.0;      
      di = floor(di * floatrange) / floatrange;
      x[i] = std::complex<octuple>(tmblas::type_conv<octuple, double>(dr),
				     tmblas::type_conv<octuple, double>(di));
    }
  }
  else {
    for (int64_t i = 0; i < size; ++i) {
      double m = m_const;
      double dr, di;
      octuple xr(0), xi(0);
      for (int j = 0; j < 4; j++, m *= m_const) {
	dr = lrand48() * m;
	di = lrand48() * m;
	octuple odr(dr);
	octuple odi(di);
	tmblas::mixedp_add<octuple, octuple, octuple>(xr, xr, odr);
	tmblas::mixedp_add<octuple, octuple, octuple>(xi, xi, odi);      
      }
      octuple xxr(-1), xxi(-1);
      tmblas::mixedp_madd<octuple, octuple, octuple>(xxr, xr, octuple(2));
      tmblas::mixedp_madd<octuple, octuple, octuple>(xxi, xi, octuple(2));    
      x[i] = std::complex<octuple>(xxr, xxi);
    }
  }
}
*/

typedef int64_t blas_int;


// -----------------------------------------------------------------------------
void lapack_potrf(  char uplo, int64_t n,
                    float *A, int64_t lda,
                    int64_t *info )
{
#ifdef INTEL_MKL_TEST
  blas_int n_ = (blas_int) n;
  blas_int lda_ = (blas_int) lda;
  LAPACKE_spotrf(LAPACK_COL_MAJOR, uplo, n_, A, lda_);
#else
    int n_ = (int) n;
    int lda_ = (int) lda;
    int *info_ = (int *) info;
    spotrf_(&uplo, &n_, A, &lda_, info_);
#endif
}

void lapack_potrf(  char uplo, int64_t n,
                    double *A, int64_t lda,
                    int64_t *info )
{
#ifdef INTEL_MKL_TEST
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    LAPACKE_dpotrf(LAPACK_COL_MAJOR,  uplo, n_, A, lda_);
#else  
    int n_ = (int) n;
    int lda_ = (int) lda;
    int *info_ = (int *) info;
    dpotrf_(&uplo, &n_, A, &lda_, info_);
#endif
}

void lapack_potrf(  char uplo, int64_t n,
                    std::complex<float> *A, int64_t lda,
                    int64_t *info )
{
#ifdef INTEL_MKL_TEST
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    LAPACKE_cpotrf(LAPACK_COL_MAJOR,  uplo, n_, (lapack_complex_float *)&A[0], lda_);
#else
    int n_ = (int) n;
    int lda_ = (int) lda;
    int *info_ = (int *) info;
    spotrf_(&uplo, &n_, (float *)A, &lda_, info_);
#endif
}

void lapack_potrf(  char uplo, int64_t n,
                    std::complex<double> *A, int64_t lda,
                    int64_t *info )
{
#ifdef INTEL_MKL_TEST
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    LAPACKE_zpotrf(LAPACK_COL_MAJOR,  uplo, n_, (lapack_complex_double *)&A[0], lda_);
#else
    int n_ = (int) n;
    int lda_ = (int) lda;
    int *info_ = (int *) info;
    dpotrf_(&uplo, &n_, (double *)A, &lda_, info_);
#endif
}
