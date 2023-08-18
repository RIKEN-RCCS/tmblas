//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <math.h>
#include <vector>
#include <list>
#include <cstdio>
#include <cstring>
#include "tmblas.hpp"
#ifdef INTEL_MKL_TEST
#include <mkl.h>
#endif

#define FLOATRANGE  1048576.0 // 2^20

struct csr_matrix {
  int nrow, nnz;
  int *ptrow, *indcol;
  int *ptrow1, *indcol1;  
  double *coefs;
};

void CSR_gemv(char const trans, double const alpha,
	      csr_matrix const a, double *x, double const beta, double *y)
{
  int nrow = a.nrow;  
  double *z = new double [nrow];
#ifdef INTEL_MKL_TEST
  mkl_dcsrgemv(&trans, &(a.nrow), &(a.coefs[0]), &(a.ptrow1[0]),
	       &(a.indcol1[0]),
	       x, z);
#else  
  if (trans == 'N' || trans == 'n') {
    for (int i = 0; i < nrow; i++) {
     z[i] = 0.0;
      for (int k = a.ptrow[i]; k < a.ptrow[i + 1]; k++) {
	int j = a.indcol[k];
	z[i] += a.coefs[k] * x[j];
      }
    }
  }
  else {
    for (int i = 0; i < nrow; i++) {
      z[i] = 0.0;
    }
    for (int i = 0; i < nrow; i++) {    
      for (int k = a.ptrow[i]; k < a.ptrow[i + 1]; k++) {
	int j = a.indcol[k];
	z[j] += a.coefs[k] * x[i];
      }
    }
  }
#endif
  cblas_dscal(nrow, beta, y, 1);
  cblas_daxpy(nrow, alpha, z, 1, y, 1);
  delete [] z;
}

void CSR_symv(bool upper, double const alpha,
	      csr_matrix const a, double *x, double const beta, double *y)
{
  int nrow = a.nrow;  
  double *z = new double [nrow];
#ifdef INTEL_MKL_TEST
  const char trans = 'n';
  mkl_dcsrgemv(&trans, &(a.nrow), &(a.coefs[0]), &(a.ptrow1[0]),
	       &(a.indcol1[0]),
	       x, z);
#else  
  if (upper) {
    for (int i = 0; i < nrow; i++) {
      z[i] = 0.0;
    }
    for (int i = 0; i < nrow; i++) {    
     {
       int k = a.ptrow[i];
       int j = a.indcol[k];
       z[i] += a.coefs[k] * x[j];
     }
      for (int k = a.ptrow[i] + 1; k < a.ptrow[i + 1]; k++) {
	int j = a.indcol[k];
	z[i] += a.coefs[k] * x[j];
	z[j] += a.coefs[k] * x[i];	
      }
    }
  }
  else {
    for (int i = 0; i < nrow; i++) {
      z[i] = 0.0;
    }
    for (int i = 0; i < nrow; i++) {    
      for (int k = a.ptrow[i]; k < a.ptrow[i + 1] - 1; k++) {
	int j = a.indcol[k];
	z[j] += a.coefs[k] * x[i];
	z[i] += a.coefs[k] * x[j];	
      }
      {
	int k = a.ptrow[i + 1] - 1;
	int j = a.indcol[k];
	z[i] += a.coefs[k] * x[j];
      }
    }
  }
#endif
  cblas_dscal(nrow, beta, y, 1);
  cblas_daxpy(nrow, alpha, z, 1, y, 1);
  delete [] z;
}

void generate_CSR(std::list<int>* ind_cols_tmp, std::list<double>* val_tmp, 
		  int nrow, int nnz, 
		  int *irow, int *jcol, double* val)
{
  for (int i = 0; i < nnz; i++) {
    const int ii = irow[i];
    const int jj = jcol[i];
    if (ind_cols_tmp[ii].empty()) {
      ind_cols_tmp[ii].push_back(jj);
      val_tmp[ii].push_back(val[i]);
    }
    else {
      if (ind_cols_tmp[ii].back() < jj) {
	ind_cols_tmp[ii].push_back(jj);
	val_tmp[ii].push_back(val[i]);
      }
      else {
	std::list<double>::iterator iv = val_tmp[ii].begin();
	std::list<int>::iterator it = ind_cols_tmp[ii].begin();
	for ( ; it != ind_cols_tmp[ii].end(); ++it, ++iv) {
	  if (*it == jj) {
	      break;
	  }
	  if (*it > jj) {
	    ind_cols_tmp[ii].insert(it, jj);
	    val_tmp[ii].insert(iv, val[i]);
	    break;
	  }
	}
      }
    }
  }
}
int generate_symCSR(std::list<int>* ind_cols_tmp, std::list<double>* val_tmp, 
		     int nrow, int nnz, 
		     int *irow, int *jcol, double* val, bool upper)
{
  int nnz1 = 0;
  for (int i = 0; i < nnz; i++) {
    const int ii = irow[i];
    const int jj = jcol[i];
    if (upper) {
      if (ii > jj) {
	continue;
      }
    }
    else {
      if (ii < jj) {
	continue;
      }
    }
    if (ind_cols_tmp[ii].empty()) {
      ind_cols_tmp[ii].push_back(jj);
      val_tmp[ii].push_back(val[i]);
      nnz1++;
    }
    else {
      if (ind_cols_tmp[ii].back() < jj) {
	ind_cols_tmp[ii].push_back(jj);
	val_tmp[ii].push_back(val[i]);
	nnz1++;
      }
      else {
	std::list<double>::iterator iv = val_tmp[ii].begin();
	std::list<int>::iterator it = ind_cols_tmp[ii].begin();
	for ( ; it != ind_cols_tmp[ii].end(); ++it, ++iv) {
	  if (*it == jj) {
	      break;
	  }
	  if (*it > jj) {
	    ind_cols_tmp[ii].insert(it, jj);
	    val_tmp[ii].insert(iv, val[i]);
	    break;
	  }
	}
	nnz1++;
      }
    }
  } // loop : i
  if (upper) {
    for (int i = 0; i < nrow; i++) {
      if (i != ind_cols_tmp[i].front()) {
	ind_cols_tmp[i].push_front(i);
	val_tmp[i].push_front(0.0);
	nnz1++;
      }
    }
 }
  else {
    for (int i = 0; i < nrow; i++) {
      if (i != ind_cols_tmp[i].back()) {
	ind_cols_tmp[i].push_back(i);
	val_tmp[i].push_back(0.0);
	nnz1++;
      }
    }
  }
  
  return nnz1;
}

int main(int argc, char **argv)
{
  int itmp, jtmp, ktmp;
  char trans[256];
  char fname[256];
  char buf[1024];
  int nrow, nnz;
  FILE *fp;
  csr_matrix a;
  bool flaggen = true;
  bool notrans = true;
  bool upper = false;
  if (argc < 3) {
    fprintf(stderr,
 	    "GMRES-ASM-MPI [data file] [trans]");
    exit(-1);
  }    
  strcpy(fname, argv[1]);
  strcpy(trans, argv[2]);
  if (trans[0] == 'N' || trans[0] == 'n' ) {
    flaggen = true;
    notrans = true;
  }
  else if (trans[0] == 'T' || trans[0] == 't') {
    flaggen = true;
    notrans = false;
  }
  else if (trans[0] == 'U' || trans[0] == 'u') {
    flaggen = false;
    upper = true;
  }
  else if (trans[0] == 'L' || trans[0] == 'l') {
    flaggen = false;
    upper = false;
  }
  if ((fp = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "fail to open %s\n", fname);
  }
  fgets(buf, 256, fp);

  while (1) {
    fgets(buf, 256, fp);
    if (buf[0] != '%') {
      sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
      nrow = itmp;
      nnz = ktmp;
      break;
    }
  }
  std::vector<int> irow(nnz);
  std::vector<int> jcol(nnz);
  std::vector<double>  val(nnz);
  {
    int ii = 0;
    int itmp, jtmp;
    float vtmp;
    for (int i = 0; i < nnz; i++) {
      fscanf(fp, "%d\t%d\t%f", &itmp, &jtmp, &vtmp);
	irow[ii] = itmp - 1; // zero based
	jcol[ii] = jtmp - 1; // zero based
	val[ii] = (double)vtmp;
	ii++;
    }
    fprintf(stderr, "%d\n", ii);
    nnz = ii;
  }

  fclose (fp);

  std::vector<std::list<int> > ind_cols_tmp2(nrow);
  std::vector<std::list<double> > val_tmp2(nrow);

  if (flaggen) {
    generate_CSR(&ind_cols_tmp2[0], &val_tmp2[0], 
		 nrow, nnz,
		 &irow[0], &jcol[0], &val[0]);
  }
  else {
    nnz = generate_symCSR(&ind_cols_tmp2[0], &val_tmp2[0], 
			  nrow, nnz,
			  &irow[0], &jcol[0], &val[0], upper);
    
  }
  a.nrow = nrow;
  a.nnz = nnz;
  a.ptrow = new int[nrow + 1];
  a.ptrow1 = new int[nrow + 1];  
  a.indcol = new int[nnz];
  a.indcol1 = new int[nnz];
  a.coefs = new double[nnz];
  {  
    int k = 0;
    a.ptrow[0] = 0;
    for (int i = 0; i < nrow; i++) {
      std::list<int>::iterator jt = ind_cols_tmp2[i].begin();
      std::list<double>::iterator jv = val_tmp2[i].begin();	
      for ( ; jt != ind_cols_tmp2[i].end(); ++jt, ++jv) {
	a.indcol[k] = (*jt);
	a.coefs[k] = floor((*jv) * FLOATRANGE)/ FLOATRANGE;
	k++;
      }
      a.ptrow[i + 1] = k;
    } // loop : i
  }
  for (int i = 0; i < (nrow + 1); i++) {
    a.ptrow1[i] = a.ptrow[i] + 1;
  }
  for (int i = 0; i < nnz; i++) {
    a.indcol1[i] = a.indcol[i] + 1;
  }
    
  std::vector<double> rhs(nrow * 2), exact(nrow * 2), y(nrow * 2);
  for (int i = 0; i < nrow * 2; i++) {
    exact[i] = (double)(i % 100);
    y[i] = rhs[i] =  floor(exact[i] / 100.0 * FLOATRANGE) / FLOATRANGE;
  }

  //  double alpha = 3.1415, beta = 2.718;
  double alpha = 3.1415, beta = 2.718;  
  alpha = floor(alpha * FLOATRANGE) / FLOATRANGE;
  beta = floor(beta * FLOATRANGE) / FLOATRANGE;  
  
  if (flaggen) {
    fprintf(stderr, "%s %d : general CSR gemv\n", __FILE__, __LINE__);
    CSR_gemv(trans[0], alpha, a, &exact[0], beta, &rhs[0]);    
    tmblas::csrgemv<double, double, double>(
	    (notrans ? blas::Op::NoTrans : blas::Op::Trans),
					    nrow, alpha,
					    a.coefs, a.ptrow, a.indcol,
					    &exact[0], beta, &y[0]);
  }
  else {
    fprintf(stderr, "%s %d : symmetric CSR gemv\n", __FILE__, __LINE__);    
    CSR_symv(upper, alpha, a, &exact[0], beta, &rhs[0]);        
    tmblas::csrsymv<double, double, double>((upper ? blas::Uplo::Upper : blas::Uplo::Lower),
					    nrow, alpha,
					    a.coefs, a.ptrow, a.indcol,
					    &exact[0], beta, &y[0]);
  }
  for (int i = 0; i < nrow; i++) {
    if (fabs (y[i] - rhs[i]) != 0.0) {
      fprintf(stderr, "%d %g %g %g\n", i, y[i], rhs[i], y[i] - rhs[i]);      
    }
  }
  for (int i = 0; i < nrow * 2; i++) {
    exact[i] = (double)(i % 100);
    y[i] = floor(exact[i] / 100.0 * FLOATRANGE) /FLOATRANGE;
  }
  if (flaggen) {
    fprintf(stderr, "%s %d : general CSR gemm\n", __FILE__, __LINE__);    
    CSR_gemv(trans[0], alpha, a, &exact[nrow], beta, &rhs[nrow]);  
    tmblas::csrgemm<double, double, double>(
	    (notrans ? blas::Op::NoTrans : blas::Op::Trans),
					    nrow, 2, alpha,
					    a.coefs, a.ptrow, a.indcol,
					    &exact[0], nrow, beta, &y[0], nrow);
  }
  else {
    fprintf(stderr, "%s %d : symmetric CSR gemm\n", __FILE__, __LINE__);        
    CSR_symv(upper, alpha, a, &exact[nrow], beta, &rhs[nrow]);        
    tmblas::csrsymm<double, double, double>((upper ? blas::Uplo::Upper : blas::Uplo::Lower),
					    nrow, 2, alpha,
					    a.coefs, a.ptrow, a.indcol,
					    &exact[0], nrow, beta, &y[0], nrow);
  }
  for (int i = 0; i < nrow * 2; i++) {
    if (fabs (y[i] - rhs[i]) != 0.0) {
      fprintf(stderr, "%d %g %g %g\n", i, y[i], rhs[i], y[i] - rhs[i]);
    }
  }
}
