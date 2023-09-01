//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "blas/flops.hh"
#include "tmblastest.hh"

#define DIM 100
typedef double Tbase;
typedef double RA;
typedef double RB;
typedef double RC;
typedef double TA;
typedef double TB;
typedef double TC;
typedef double TD;

// -----------------------------------------------------------------------------
int main ()
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RB, RC >;
    using scalar_tt = blas::scalar_type< TA, TB, TC >;

    // get & mark input values
    blas::Layout layout = blas::Layout::ColMajor;
    blas::Op transA = blas::Op::NoTrans;
    blas::Op transB = blas::Op::NoTrans;
    Tbase alpha = Tbase(1);
    Tbase beta = Tbase(0);
    scalar_t alphar = tmblas::type_conv<scalar_t,  Tbase>(alpha);
    scalar_tt alphat = tmblas::type_conv<scalar_tt, Tbase>(alpha);
    scalar_t betar = tmblas::type_conv<scalar_t,  Tbase>(beta);
    scalar_tt betat = tmblas::type_conv<scalar_tt, Tbase>(beta);
    
    int64_t m       = DIM;
    int64_t n       = DIM;
    int64_t k       = DIM;
    int64_t align   = 32;
    
    // setup
    int64_t Am = (transA == Op::NoTrans ? m : k);
    int64_t An = (transA == Op::NoTrans ? k : m);
    int64_t Bm = (transB == Op::NoTrans ? k : n);
    int64_t Bn = (transB == Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    int64_t ldc = roundup( Cm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    size_t size_C = size_t(ldc)*Cn;
    Tbase* A    = new Tbase[ size_A ];
    Tbase* B    = new Tbase[ size_B ];
    Tbase* C    = new Tbase[ size_C ];
    RA* Ar   = new RA[ size_A ];
    RB* Br   = new RB[ size_B ];
    RC* Cr   = new RC[ size_C ];
    TA* At   = new TA[ size_A ];
    TB* Bt   = new TB[ size_B ];
    TC* Ct   = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_A, A, false);
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_B, B, false);
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_C, C, false);

    // norms for error check
    for (size_t i = 0; i < size_A; i++) {
      At[i] = tmblas::type_conv<TA, Tbase>(A[i]);
      Ar[i] = tmblas::type_conv<RA, Tbase>(A[i]);      
    }
    for (size_t i = 0; i < size_B; i++) {
      Bt[i] = tmblas::type_conv<TB, Tbase>(B[i]);
      Br[i] = tmblas::type_conv<RB, Tbase>(B[i]);      
    }
    for (size_t i = 0; i < size_C; i++) {
      Ct[i] = tmblas::type_conv<TC, Tbase>(C[i]);
      Cr[i] = tmblas::type_conv<RC, Tbase>(C[i]);      
    }

    // run test
    tmblas::gemm<TA, TB, TC, TD>( transA, transB, m, n, k,
			      alphat, At, lda, Bt, ldb, betat, Ct, ldc );

    // run reference
	blas::gemm<RA, RB, RC>( layout, transA, transB,
                  m, n, k, alphar, Ar, lda, Br, ldb, betar, Cr, ldc );

    // check error compared to reference
    double errormax, errorl2;
	check_diff<RC, TC>(Cm, Cn, Cr, Ct, ldc, &errormax, &errorl2);
    std::cout << "errormax: " << errormax << std::endl;
    std::cout << "errorl2: " << errorl2 << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Ar;
    delete[] Br;
    delete[] Cr;
    delete[] At;
    delete[] Bt;
    delete[] Ct;

	return 0;
}

