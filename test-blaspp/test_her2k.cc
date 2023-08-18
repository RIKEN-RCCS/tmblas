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

// -----------------------------------------------------------------------------
template <typename Tbase, typename RA, typename RB, typename RC, typename TA, typename TB, typename TC>
void test_her2k_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RC >;
    using scalar_tt = blas::scalar_type< TA, TC >;
    using real_t   = blas::real_type< scalar_t >;
    using real_tt  = blas::real_type< scalar_tt >;
    using Tbasereal   = blas::real_type< Tbase >;

    bool trunc = true;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    blas::Uplo uplo = params.uplo();
    Tbase alpha(1);
    Tbasereal beta;
    //    tmblas_setscal<Tbase>(alpha, params.alpha(), params.alphai());
    tmblas_setscal<Tbasereal>(beta,  params.beta());
    if (trunc) {
      alpha = tmblas_trunc<Tbase>(alpha);
      beta = tmblas_trunc<Tbasereal>(beta);
    }
    scalar_t alphar  = tmblas::type_conv<scalar_t,  Tbase>(alpha);
    scalar_tt alphat = tmblas::type_conv<scalar_tt, Tbase>(alpha);
    real_t betar   = tmblas::type_conv<real_t,  Tbasereal>(beta);
    real_tt betat  = tmblas::type_conv<real_tt, Tbasereal>(beta);
    
    int64_t n       = params.dim.n();
    int64_t k       = params.dim.k();
    int64_t align   = params.align();
//    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // setup
    int64_t Am = (trans == Op::NoTrans ? n : k);
    int64_t An = (trans == Op::NoTrans ? k : n);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Am, align );
    int64_t ldc = roundup(  n, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*An;
    size_t size_C = size_t(ldc)*n;
    Tbase* A    = new Tbase[ size_A ];
    Tbase* B    = new Tbase[ size_B ];
    Tbase* C    = new Tbase[ size_C ];
    
    RA* Ar    = new RA[ size_A ];
    RB* Br    = new RB[ size_B ];
    RC* Cr    = new RC[ size_C ];
    TA* At   = new TA[ size_A ];
    TB* Bt   = new TB[ size_B ];
    TC* Ct   = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv( idist, iseed, size_A, A, trunc );
    tmblas_larnv( idist, iseed, size_B, B, trunc );
    tmblas_larnv( idist, iseed, size_C, C, trunc );
#if 1
    for (int64_t i = 0; i < n; i++) {
      C[i * (n + 1)] = std::complex<blas::real_type<Tbase> >(C[i * (n + 1)].real(), blas::real_type<Tbase>(0));
    }
#endif
    //    lapack_lacpy( "g", n, n, C, ldc, Cref, ldc );

    // norms for error check
#if 0
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Am, An, B, ldb, work );
    real_t Cnorm = lapack_lansy( "f", uplo2str(uplo), n, C, ldc, work );
#endif
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
#if 0
    // test error exits
    assert_throw( tmblas::her2k( Uplo(0), trans,  n,  k, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::her2k( uplo,    Op(0),  n,  k, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::her2k( uplo,    trans, -1,  k, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::her2k( uplo,    trans,  n, -1, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::her2k( uplo, Op::NoTrans,   n, k, alphat, At, n-1, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::her2k( uplo, Op::Trans,     n, k, alphat, At, k-1, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::her2k( uplo, Op::ConjTrans, n, k, alphat, At, k-1, Bt, ldb, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::her2k( uplo, Op::NoTrans,   n, k, alphat, At, lda, Bt, n-1, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::her2k( uplo, Op::Trans,     n, k, alphat, At, lda, Bt, k-1, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::her2k( uplo, Op::ConjTrans, n, k, alphat, At, lda, Bt, k-1, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::her2k( uplo,    trans,  n,  k, alphat, At, lda, Bt, ldb, betat, Ct, n-1 ), blas::Error );

    if (blas::is_complex<scalar_t>::value) {
        // complex her2k doesn't allow Trans, only ConjTrans
        assert_throw( tmblas::her2k( uplo, Op::Trans, n, k, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );
    }

    if (verbose >= 1) {
        printf( "\n"
                "uplo %c, trans %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bn=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C  n=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                uplo2char(uplo), op2char(trans),
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Am ), llong( An ), llong( ldb ), llong( size_B ), Bnorm,
                llong( n ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e;  %% beta real\n",
                real(alpha), imag(alpha),
                beta );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Am, An, B, ldb );
        printf( "C = "    ); print_matrix(  n,  n, C, ldc );
    }
#endif
    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::her2k<TA, TB, TC>( uplo, trans, n, k,
                 alphat, At, lda, Bt, ldb, betat, Ct, ldc );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::her2k( n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

#if 0
    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( n, n, C, ldc );
    }
#endif

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
	blas::her2k<RA, RB, RC>(layout, uplo, trans, 
                     n, k, alphar, Ar, lda, Br, ldb, betar, Cr, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;
#if 0
        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( n, n, Cref, ldc );
        }
#endif
        // check error compared to reference
        double errormax, errorl2;	
        bool okay = true;
	//       check_gemm( Cm, Cn, An, alpha, beta, Anorm, Bnorm, Cnorm,
	//                    Cref, ldc, C, ldc, verbose, &error, &okay );
	check_diff<RC, TC>(n, n, Cr, Ct, ldc, &errormax, &errorl2, uplo);
	if (trunc) {
	  if ((errormax != 0.0) || (errorl2 != 0.0)){
	    okay = false;
	  }
	}	
        params.error() = errormax;
        params.error2() = errorl2;		   	
        params.okay() = okay;	
    }
    delete[] A;
    delete[] B;        
    delete[] C;    
    delete[] Ar;
    delete[] Br;
    delete[] Cr;
    delete[] At;
    delete[] Bt;
    delete[] Ct;
}

// -----------------------------------------------------------------------------
void test_her2k( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
        case testsweeper::DataType::Double:
        case testsweeper::DataType::Quadruple:
        case testsweeper::DataType::Octuple:

        case testsweeper::DataType::SingleComplex:
            test_her2k_work< std::complex<float>,
			     std::complex<float>, std::complex<float>, std::complex<float>,
			     std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_her2k_work< std::complex<double>,
			     std::complex<double>, std::complex<double>, std::complex<double>,
			     std::complex<double>, std::complex<double>, std::complex<double> >( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
	  test_her2k_work< std::complex<double>,
			   std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>,
			   std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >( params, run );
            break;

        case testsweeper::DataType::OctupleComplex:
	  test_her2k_work< std::complex<double>,
			   std::complex<octuple>, std::complex<octuple>, std::complex<octuple>,
			   std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
