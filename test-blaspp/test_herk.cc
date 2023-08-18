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
template <typename Tbase, typename RA, typename RC, typename TA, typename TC>
void test_herk_work( Params& params, bool run )
{
    using namespace testsweeper;
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
    Tbasereal alpha, beta;
    tmblas_setscal<Tbasereal>(alpha, params.alpha());
    tmblas_setscal<Tbasereal>(beta,  params.beta());
    if (trunc) {
      alpha = tmblas_trunc<Tbasereal>(alpha);
      beta = tmblas_trunc<Tbasereal>(beta);
    }
    real_t alphar  = tmblas::type_conv<real_t,  Tbasereal>(alpha);
    real_tt alphat = tmblas::type_conv<real_tt, Tbasereal>(alpha);
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
    int64_t ldc = roundup(  n, align );
    size_t size_A = size_t(lda)*An;
    size_t size_C = size_t(ldc)*n;

    Tbase* A    = new Tbase[ size_A ];
    Tbase* C    = new Tbase[ size_C ];
							   
    RA* Ar   = new RA[ size_A ];
    RC* Cr   = new RC[ size_C ];
    TA* At   = new TA[ size_A ];
    TC* Ct   = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_A, A, trunc );
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_C, C, trunc );
#if 0
    for (int64_t i = 0; i < n; i++) {
      C[i * (n + 1)] = std::complex<blas::real_type<Tbase> >(C[i * (n + 1)].real(), blas::real_type<Tbase>(0));
    }
#endif
    //    lapack_lacpy( "g", n, n, C, ldc, Cref, ldc );

    // norms for error check
#if 0
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Cnorm = lapack_lansy( "f", uplo2str(uplo), n, C, ldc, work );
#endif
    for (size_t i = 0; i < size_A; i++) {
      At[i] = tmblas::type_conv<TA, Tbase>(A[i]);
      Ar[i] = tmblas::type_conv<RA, Tbase>(A[i]);      
    }
    for (size_t i = 0; i < size_C; i++) {
      Ct[i] = tmblas::type_conv<TC, Tbase>(C[i]);
      Cr[i] = tmblas::type_conv<RC, Tbase>(C[i]);
    }
#if 0
    // test error exits
    assert_throw( tmblas::herk( Uplo(0), trans,  n,  k, alphat, At, lda, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::herk( uplo,    Op(0),  n,  k, alphat, At, lda, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::herk( uplo,    trans, -1,  k, alphat, At, lda, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::herk( uplo,    trans,  n, -1, alphat, At, lda, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::herk( uplo, Op::NoTrans,   n, k, alphat, At, n-1, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::herk( uplo, Op::Trans,     n, k, alphat, At, k-1, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::herk( uplo, Op::ConjTrans, n, k, alphat, At, k-1, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::herk( uplo,    trans,  n,  k, alphat, At, lda, betat, Ct, n-1 ), blas::Error );

    if (blas::is_complex<scalar_t>::value) {
        // complex herk doesn't allow Trans, only ConjTrans
        assert_throw( tmblas::herk( uplo, Op::Trans, n, k, alphat, At, lda, betat, Ct, ldc ), blas::Error );
    }

    if (verbose >= 1) {
        printf( "\n"
                "uplo %c, trans %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "C  n=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                uplo2char(uplo), op2char(trans),
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( n ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e; beta = %.4e;  %% real\n", alpha, beta );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "C = "    ); print_matrix(  n,  n, C, ldc );
    }
#endif
    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::herk<TA, TC>( uplo, trans, n, k,
                alphat, At, lda, betat, Ct, ldc );
    time = get_wtime() - time;
//    std::cerr << __LINE__ << " " << Ct[0] << std::endl;
    double gflop = blas::Gflop< scalar_t >::herk( n, k );
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
	blas::herk<RA, RC>( layout, uplo, trans,
                    n, k, alphar, Ar, lda, betar, Cr, ldc );
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
    delete[] C;
    delete[] Ar;
    delete[] Cr;
    delete[] At;
    delete[] Ct;
}

// -----------------------------------------------------------------------------
void test_herk( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
        case testsweeper::DataType::Double:
        case testsweeper::DataType::Quadruple:
        case testsweeper::DataType::Octuple:

        case testsweeper::DataType::SingleComplex:
            test_herk_work< std::complex<float>,
			    std::complex<float>, std::complex<float>,
			    std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_herk_work< std::complex<double>,
			    std::complex<double>, std::complex<double>,
			    std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_herk_work< std::complex<double>,
			    std::complex<quadruple>, std::complex<quadruple>,
			    std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

	    case testsweeper::DataType::OctupleComplex:
            test_herk_work< std::complex<double>,
			    std::complex<octuple>, std::complex<octuple>,
			    std::complex<octuple>, std::complex<octuple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
