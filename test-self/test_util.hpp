//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef _TEST_UTIL_HPP
# define _TEST_UTIL_HPP
#include "tmblas.hpp"
#include <string>
#include <cstring>
#include <cmath>
#include <stdio.h>
#define FLOATRANGE 1048576.0 // 2^20

template<typename Ta, typename Td = Ta>
void print_result_csv(std::string msg, const int n, const Ta *X, const Td *Xref);

template<typename Ta, typename Td>
void print_result_csv(std::string msg, const int n, const Ta *X, const Td *Xref);

template<typename Ta, typename Td>
void print_result_csv(std::string msg, const int n, const Ta *X, const Td *Xref) {
  typedef blas::real_type<Ta> real_ta;

  real_ta maxdiff(0);
  real_ta maxval(0);
  int exmatch = 0;
  for (int i=0 ; i<n ; ++i) {
    real_ta tmp = tmblas::abs1<real_ta>(const_cast<Ta &>(X[i]));
    Ta Xrefa = tmblas::type_conv<Ta, Td>(Xref[i]);
    if(maxval < tmp) {
      maxval = tmp;
    }
    if(X[i] == Xrefa) {
      exmatch++;
    }
    else{
      Ta tmpdiff = X[i] - Xrefa;
      real_ta tmp = tmblas::abs1<real_ta>(const_cast<Ta &>(tmpdiff));
      //printf("foobar %8.3e %8.3e\n",X[i],Xref[i]);
      if(maxdiff < tmp) {
        maxdiff = tmp;
      }
    }
  }
  printf("%s,%i,%i,%s,%s\n",msg.c_str(),n,exmatch,
	 tmblas::tostring<real_ta>(maxval).c_str(),
	 tmblas::tostring<real_ta>(maxdiff/maxval).c_str());
}

/*
template<>
inline void print_result_csv(std::string msg, const int n, const int *X, const int *Xref) {
  int maxdiff = 0;
  int exmatch = 0;
  for (int i=0 ; i<n ; ++i) {
    if(X[i] == Xref[i]) {
      exmatch++;
    }
    else{
      if(maxdiff<tmblas::abs1(X[i]-Xref[i])) {
        maxdiff = tmblas::abs1(X[i]-Xref[i]);
      }
    }
  }
  printf("%s,%i,%i,%i\n",msg.c_str(),n,exmatch,maxdiff);
}
*/

#ifdef DD_REAL
template<typename Ta>
void print_result_csv_qd(std::string msg, const int n, const octuple *Xoct, const Ta *Xa) {
  octuple maxdiff(0);
  octuple maxval(0);
  octuple otmp0, otmp1, otmp0a;
  int exmatch = 0;
  for (int i=0 ; i<n ; ++i) {
    otmp0 = tmblas::type_conv<octuple, Ta>(Xa[i]);
    otmp0a = fabs(otmp0);
    if (maxval < otmp0a) {
      maxval = otmp0a;
    }
    if(Xoct[i] == otmp0) {
      exmatch++;
    }
    else{
      otmp1 = fabs(Xoct[i] - otmp0); // with upcast
      if(maxdiff < otmp1) {
        maxdiff = otmp1;
      }
    }
  }
  printf("%s,%s\n",msg.c_str(),tmblas::tostring<octuple>(maxdiff/maxval).c_str());
}

template
void print_result_csv_qd<quadruple>(std::string msg, const int n, const octuple *Xoct, const quadruple *Xa);

template
void print_result_csv_qd<double>(std::string msg, const int n, const octuple *Xoct, const double *Xa);

template
void print_result_csv_qd<float>(std::string msg, const int n, const octuple *Xoct, const float *Xa);

template
void print_result_csv_qd<half>(std::string msg, const int n, const octuple *Xoct, const half *Xa);

template<typename Ta>
void print_result_csv_qd(std::string msg, const int n, const std::complex<octuple> *Xoct, const std::complex<Ta> *Xa) {
  octuple maxdiff(0);
  octuple maxval(0);
  octuple otmp0r, otmp0i, otmp0a, otmp1r, otmp1i, otmp1a;
  int exmatch = 0;
  for (int i=0 ; i<n ; ++i) {
    otmp0r = tmblas::type_conv<octuple, Ta>(Xa[i].real());
    otmp0i = tmblas::type_conv<octuple, Ta>(Xa[i].imag());
    otmp0a = fabs(otmp0r) + fabs(otmp0i);
    if (maxval < otmp0a) {
      maxval = otmp0a;
    }
    if(Xoct[i].real() == otmp0r && Xoct[i].imag() == otmp0i) {
      exmatch++;
    }
    else{
      otmp1a = fabs(Xoct[i].real() - otmp0r) + fabs(Xoct[i].imag() - otmp0i);
      if(maxdiff < otmp1a) {
        maxdiff = otmp1a;
      }
    }
  }
  printf("%s,%s\n",msg.c_str(),tmblas::tostring<octuple>(maxdiff/maxval).c_str());
}

template
void print_result_csv_qd<quadruple>(std::string msg, const int n, const std::complex<octuple> *Xoct, const std::complex<quadruple> *Xa);

template
void print_result_csv_qd<double>(std::string msg, const int n, const std::complex<octuple> *Xoct, const std::complex<double> *Xa);

template
void print_result_csv_qd<float>(std::string msg, const int n, const std::complex<octuple> *Xoct, const std::complex<float> *Xa);

template
void print_result_csv_qd<half>(std::string msg, const int n, const std::complex<octuple> *Xoct, const std::complex<half> *Xa);

template<typename T>
T get_diff_wrt_oct(T tval, octuple oval) {
  octuple t = tmblas::type_conv<octuple, T>(tval);
  octuple w = t - oval;
  return tmblas::type_conv<T, octuple>(tmblas::abs1(w));
}

template
quadruple get_diff_wrt_oct(quadruple qval, octuple oval);
#endif

template<typename Ta>
void print_result_csv_2d(std::string msg, blas::Uplo uplo, const int n, const int m, const Ta *X, const Ta *Xref) {
  typedef blas::real_type<Ta> real_ta;
  real_ta maxdiff(0);
  real_ta maxval(0);
  int exmatch = 0;
  int ncand = 0;
  for (int j=0 ; j<n ; ++j) {
    if (uplo == blas::Uplo::General) {
      for (int i=0 ; i<m ; ++i) {
	real_ta Xa = tmblas::abs1<real_ta>(const_cast<Ta &>(X[i + j * m]));
        if(maxval < Xa) {
          maxval = Xa;
        }
        if(X[i + j*m] == Xref[i + j*m]) {
          exmatch++;
        }
        else{
	  Ta difftmp = X[i + j*m]-Xref[i + j*m];
	  real_ta diffval = tmblas::abs1<real_ta>(const_cast<Ta &>(difftmp));
          if(maxdiff < diffval) {
            maxdiff = diffval;
          }
        }
        ncand += 1;
      }
    }
    // upper -> row index <= column index
    else if (uplo == blas::Uplo::Upper) {
      for (int i=0 ; i<=j ; ++i) {
	real_ta Xa = tmblas::abs1<real_ta>(const_cast<Ta &>(X[i + j * m]));
        if(maxval < Xa) {
          maxval = Xa;
        }
        if(X[i + j*m] == Xref[i + j*m]) {
          exmatch++;
        }
        else{
	  Ta difftmp = X[i + j*m]-Xref[i + j*m];
	  real_ta diffval = tmblas::abs1<real_ta>(const_cast<Ta &>(difftmp));
          if(maxdiff< diffval) {
            maxdiff = diffval;
          }
        }
        ncand += 1;
      }
    }
    // lower -> row index >= column index
    else {
      for (int i=j ; i<m ; ++i) {
	real_ta Xa = tmblas::abs1<real_ta>(const_cast<Ta &>(X[i + j*m]));
        if(maxval < Xa) {
          maxval = Xa;
        }
        if(X[i + j*m] == Xref[i + j*m]) {
          exmatch++;
        }
	else {
	  Ta difftmp = X[i + j*m]-Xref[i + j*m];
	  real_ta diffval = tmblas::abs1<real_ta>(const_cast<Ta &>(difftmp));
          if(maxdiff< diffval) {
            maxdiff = diffval;
          }
        }
        ncand += 1;
      }
    }
  }
  printf("%s,%i,%i,%8.3e,%8.3e\n",msg.c_str(),ncand,exmatch,maxval,maxdiff/maxval);
//  printf("number of elements which exactly matched to each other %i\n", exmatch);
//  printf("max normalized diff %8.3e\n", maxdiff/maxval);
}

template<typename Ta>
void print_result(const int n, const Ta *X, const Ta *Xref);

template<typename Ta>
void print_result(const int n, const Ta *X, const Ta *Xref) {
  double maxdiff = 0;
  double maxval = 0;
  int exmatch = 0;
  for (int i=0 ; i<n ; ++i) {
#ifdef DEBUG_TMBLAS
    printf("tmblas blas++ %8.3e %8.3e\n",X[i],Xref[i]);
#endif
    if(maxval<tmblas::abs1(X[i])) {
      maxval = tmblas::abs1(X[i]);
    }
    if(X[i] == Xref[i]) {
      exmatch++;
    }
    else{
      if(maxdiff<tmblas::abs1(X[i]-Xref[i])) {
        maxdiff = tmblas::abs1(X[i]-Xref[i]);
      }
    }
  }
  printf("number of elements which exactly matched to each other %i\n", exmatch);
  printf("max normalized diff %8.3e\n", maxdiff/maxval);
}

template<typename Tc>
void print_result_2d(const int m, const int n, const Tc *C, const Tc *Cref, const blas::Uplo uplo);

template<typename Tc>
void print_result_2d(const int m, const int n, const Tc *C, const Tc *Cref, const blas::Uplo uplo) {
  int passed = 0;
//  double maxdiff = double(0.0);
//  double maxval = 0;
  blas::real_type<Tc> maxdiff = blas::real_type<Tc>(0);
  blas::real_type<Tc> maxval = blas::real_type<Tc>(0);
  if(uplo != blas::Uplo::General && m != n) {
    printf("uplo must be blas::Uplo::General if m and n differ");
    return;
  }
  for (int j=0 ; j<n ; ++j) {
    if(uplo == blas::Uplo::General) {
      for(int i=0 ; i<m ; ++i) {
        if(maxval<tmblas::abs1(C[i+j*m])) {
          maxval = tmblas::abs1(C[i+j*m]);
        }
        if (C[i+j*m] == Cref[i+j*m]) {
            passed++;
        }
        else {
            
            if (maxdiff<tmblas::abs1(C[i+j*m]-Cref[i+j*m])) {
                maxdiff = tmblas::abs1(C[i+j*m]-Cref[i+j*m]);
            }
        }
      }
    } 
    else if (uplo == blas::Uplo::Lower) {
      for(int i=0 ; i<=j ; ++i) {
        if(maxval<tmblas::abs1(C[i+j*m])) {
          maxval = tmblas::abs1(C[i+j*m]);
        }
        if (C[i+j*m] == Cref[i+j*m]) {
            passed++;
        }
        else {
            if (maxdiff<tmblas::abs1(C[i+j*m]-Cref[i+j*m])) {
                maxdiff = tmblas::abs1(C[i+j*m]-Cref[i+j*m]);
            }
        }
      }
    } 
    else if (uplo == blas::Uplo::Upper) {
      for(int i=j ; i<n ; ++i) {
        //printf("blaspp tmblas %8.3e %8.3e\n", Cref[i+j*m],C[i+j*m]);
        if(maxval<tmblas::abs1(C[i+j*m])) {
          maxval = tmblas::abs1(C[i+j*m]);
        }
        if (C[i+j*m] == Cref[i+j*m]) {
            passed++;
        }
        else {
            if (maxdiff<tmblas::abs1(C[i+j*m]-Cref[i+j*m])) {
                maxdiff = tmblas::abs1(C[i+j*m]-Cref[i+j*m]);
            }
        }
      }
    }
  }
  printf("number of elements which exactly matched to each other %i\n", passed);
  printf("max normalized diff %8.3e\n", maxdiff/maxval);
}

template<typename Tc>
void print_result_2d(std::string msg, const int m, const int n, const Tc *C, const Tc *Cref, const blas::Op op);

template<typename Tc>
void print_result_2d(std::string msg, const int m, const int n, const Tc *C, const Tc *Cref, const blas::Op op) {
  int passed = 0;
  int ncand = 0;
//  double maxdiff = double(0.0);
//  double maxval = 0;
  typedef blas::real_type<Tc> real_tc;
  real_tc maxdiff = blas::real_type<Tc>(0);
  real_tc maxval = blas::real_type<Tc>(0);
  for (int j=0 ; j<n ; ++j) {
    for(int i=0 ; i<m ; ++i) {
      real_tc Ca = tmblas::abs1<real_tc>(const_cast<Tc &>(C[i+j*m]));
      if(maxval < Ca) {
	maxval = Ca;
      }
       
      if(op == blas::Op::NoTrans) {
        if (C[i+j*m] == Cref[i+j*m]) {
            passed++;
        }
        else {
	  Tc Cdiff = C[i+j*m] - Cref[i+j*m];
	  real_tc Cdiffa = tmblas::abs1<real_tc>(const_cast<Tc &>(Cdiff));
          if (maxdiff< Cdiffa ){
              maxdiff = Cdiffa;
          }
        }
      } else if (op == blas::Op::Trans) {
        if (C[i+j*m] == Cref[j+i*n]) {
            passed++;
        }
        else {
	  Tc Cdiff = C[i+j*m] - Cref[j+i*n];
	  real_tc Cdiffa = tmblas::abs1<real_tc>(const_cast<Tc &>(Cdiff));
          if (maxdiff< Cdiffa) {
              maxdiff = Cdiffa;
          }
        }
      } else {
        if (C[i+j*m] == tmblas::conjg(Cref[j+i*n])) {
            passed++;
        }
        else {
	  Tc Cdiff = C[i+j*m] - tmblas::conjg(Cref[j+i*n]);
	  real_tc Cdiffa = tmblas::abs1<real_tc>(const_cast<Tc &>(Cdiff));
          if (maxdiff < Cdiffa) {
	    maxdiff = Cdiffa;
          }
        }
      }
      ncand += 1;
    } 
  }
  printf("%s,%i,%i,%8.3e,%8.3e\n",msg.c_str(),ncand,passed,maxval,maxdiff/maxval);
}

template<typename T>
T get_nice_denominator() {
  int i = int(floor(drand48()*5));
  return T(std::pow(2,(i-2)));
}

template<typename T>
void fill_array(int n, T *array, bool normalized);

template<typename T>
void fill_array(int n, T *array);

template<typename T>
void fill_array(int n, T *array, bool normalized) {
  if(normalized) {
    for(int i=0 ; i<n ; ++i) {
//      array[i] = T(drand48()-0.5);
      array[i] = T(floor(FLOATRANGE * (drand48()-0.5))/512.0);
    }
  }
  else{
    for(int i=0 ; i<n ; ++i) {
      array[i] = T(drand48()-0.5);
    }
  }
}


#ifdef DD_REAL
template<>
inline void fill_array<octuple>(int n, octuple *array, bool normalized) {
  if(normalized) {
    for(int i=0 ; i<n ; ++i) {
      //array[i] = qdrand()-octuple(0.5);
      array[i] = octuple(floor(FLOATRANGE * (qdrand()-0.5))/512.0);
    }
  }
  else{
    for(int i=0 ; i<n ; ++i) {
      array[i] = octuple(qdrand()-0.5);
    }
  }
}
#endif

template<typename T>
void fill_array(int n, T *array) {
  fill_array(n, array, false);
}


template<typename T>
void fill_array(int n, std::complex<T> *array, bool normalized);

template<typename T>
void fill_array(int n, std::complex<T> *array);


template<typename T>
void fill_array(int n, std::complex<T> *array, bool normalized) {
  if(normalized) {
    for(int i=0 ; i<n ; ++i) {
      array[i].real(T(floor(FLOATRANGE * (drand48()-0.5))/512.0));
      array[i].imag(T(floor(FLOATRANGE * (drand48()-0.5))/512.0));
    }
  }
  else{
    for(int i=0 ; i<n ; ++i) {
      array[i].real(T(drand48()-0.5));
      array[i].imag(T(drand48()-0.5));
    }
  }
}

#ifdef DD_REAL
template<>
inline void fill_array(int n, std::complex<octuple> *array, bool normalized) {
  if(normalized) {
    for(int i=0 ; i<n ; ++i) {
      array[i].real(floor(qdrand()-octuple(0.5))/512.0);
      array[i].imag(floor(qdrand()-octuple(0.5))/512.0);
    }
  }
  else{
    for(int i=0 ; i<n ; ++i) {
      array[i].real(octuple(qdrand()-octuple(0.5)));
      array[i].imag(octuple(qdrand()-octuple(0.5)));
    }
  }
}
#endif

template<typename T>
void fill_array(int n, std::complex<T> *array) {
  fill_array(n, array, false);
}


template<typename T> 
T* get_array_real(int n, bool small);

template<typename T> 
T* get_array_real(int n);

template<typename T> 
T* get_array_real(int n, bool normalized){
   T *ret = new T[n];
   fill_array(n, ret, normalized);
   return ret;
}

template<typename T> 
T* get_array_real(int n) {
  return get_array_real<T>(n, false);
}

template<typename T> 
std::complex<T>* get_array_complex(int n);

template<typename T> 
std::complex<T>* get_array_complex(int n, bool small);

template<typename T> 
std::complex<T>* get_array_complex(int n, bool normalized){
   std::complex<T> *ret = new std::complex<T>[n];
   fill_array(n, ret, normalized);
   return ret;
}

template<typename T> 
std::complex<T>* get_array_complex(int n){
  return get_array_complex<T>(n, false);
}

template <typename T>
void copy_array(int n, T* const source, T* target);

template<typename T>
void copy_array(int n, T* const source, T* target) {
  for(int i=0 ; i<n ; ++i) {
    target[i] = source[i];
  }
}

template<typename Ta, typename Tb>
void conv_and_copy_array(int n, Ta* const source, Tb* target) {
  for(int i=0 ; i<n ; ++i) {
    target[i] = Tb(source[i]);
  }
}

#ifdef DD_REAL
#ifdef HALF
template<>
inline void conv_and_copy_array<octuple, half>(int n, octuple* const source, half* target) {
  for(int i=0 ; i<n ; ++i)
    target[i] = half(to_double(source[i]));
}

template<>
inline void conv_and_copy_array<std::complex<octuple>, std::complex<half>>(
            int n, std::complex<octuple>* const source, std::complex<half>* target) {
  for(int i=0 ; i<n ; ++i) {
    octuple re = source[i].real();
    octuple im = source[i].imag();
    target[i].real(half(to_double(re)));
    target[i].imag(half(to_double(im)));
  }
}
#endif

template<>
inline void conv_and_copy_array<octuple, float>(int n, octuple* const source, float* target) {
  for(int i=0 ; i<n ; ++i) {
    target[i] = float(to_double(source[i]));
  }
}

template<>
inline void conv_and_copy_array<std::complex<octuple>, std::complex<float>>(
            int n, std::complex<octuple>* const source, std::complex<float>* target) {
  for(int i=0 ; i<n ; ++i) {
    octuple re = source[i].real();
    octuple im = source[i].imag();
    target[i].real(float(to_double(re)));
    target[i].imag(float(to_double(im)));
  }
}

template<>
inline void conv_and_copy_array<octuple, double>(int n, octuple* const source, double* target) {
  for(int i=0 ; i<n ; ++i) {
    target[i] = to_double(source[i]);
  }
}

template<>
inline void conv_and_copy_array<std::complex<octuple>, std::complex<double>>(
            int n, std::complex<octuple>* const source, std::complex<double>* target) {
  for(int i=0 ; i<n ; ++i) {
    octuple re = source[i].real();
    octuple im = source[i].imag();
    target[i].real(to_double(re));
    target[i].imag(to_double(im));
  }
}

template<>
inline void conv_and_copy_array<octuple, quadruple>(int n, octuple* const source, quadruple* target) {
  for(int i=0 ; i<n ; ++i) {
    target[i] = to_dd_real(source[i]);
  }
}

template<>
inline void conv_and_copy_array<std::complex<octuple>, std::complex<quadruple>>(
            int n, std::complex<octuple>* const source, std::complex<quadruple>* target) {
  for(int i=0 ; i<n ; ++i) {
    octuple re = source[i].real();
    octuple im = source[i].imag();
    target[i].real(to_dd_real(re));
    target[i].imag(to_dd_real(im));
  }
}
#endif

#endif

