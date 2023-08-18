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
template <typename Tbase, typename RA, typename RB, typename TA, typename TB>
void test_trmm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Uplo;
    using blas::Side;
    using blas::Op;
    using blas::Layout;
    using blas::Diag;
    using scalar_t = blas::scalar_type< RA, RB >;
    using scalar_tt = blas::scalar_type< TA, TB >;

    bool trunc = true;
	
    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    blas::Op trans  = params.trans();
    blas::Diag diag = params.diag();
    Tbase alpha;
    tmblas_setscal<Tbase>(alpha, params.alpha(), params.alphai());
    if (trunc) {
      alpha = tmblas_trunc<Tbase>(alpha);
    }
    scalar_t alphar = tmblas::type_conv<scalar_t,  Tbase>(alpha);
    scalar_tt alphat = tmblas::type_conv<scalar_tt, Tbase>(alpha);

    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t align   = params.align();
    //    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // ----------
    // setup
    int64_t Am = (side == Side::Left ? m : n);
    int64_t Bm = m;
    int64_t Bn = n;
    if (layout == Layout::RowMajor)
        std::swap( Bm, Bn );
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    size_t size_A = size_t(lda)*Am;
    size_t size_B = size_t(ldb)*Bn; 
    Tbase* A    = new Tbase[ size_A ];
    Tbase* B    = new Tbase[ size_B ];
    RA* Ar    = new RA[ size_A ];
    RB* Br    = new RB[ size_B ];
    TA* At   = new TA[ size_A ];
    TB* Bt   = new TB[ size_B ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_A, A, trunc );
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_B, B, trunc );
    //    lapack_lacpy( "g", Bm, Bn, B, ldb, Bref, ldb );

    // norms for error check
#if 0
    real_t work[1];
    real_t Anorm = lapack_lantr( "f", uplo2str(uplo), diag2str(diag),
                                 Am, Am, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );
#endif
    for (size_t i = 0; i < size_A; i++) {
      At[i] = tmblas::type_conv<TA, Tbase>(A[i]);
      Ar[i] = tmblas::type_conv<RA, Tbase>(A[i]);      
    }
    for (size_t i = 0; i < size_B; i++) {
      Bt[i] = tmblas::type_conv<TB, Tbase>(B[i]);
      Br[i] = tmblas::type_conv<RB, Tbase>(B[i]);      
    }

    // test error exits
#if 0
    assert_throw( tmblas::trmm( Side(0), uplo,    trans, diag,     m,  n, alphat, At, lda, Bt, ldb ), blas::Error );
    assert_throw( tmblas::trmm( side,    Uplo(0), trans, diag,     m,  n, alphat, At, lda, Bt, ldb ), blas::Error );
    assert_throw( tmblas::trmm( side,    uplo,    Op(0), diag,     m,  n, alphat, At, lda, Bt, ldb ), blas::Error );
    assert_throw( tmblas::trmm( side,    uplo,    trans, Diag(0),  m,  n, alphat, At, lda, Bt, ldb ), blas::Error );
    assert_throw( tmblas::trmm( side,    uplo,    trans, diag,    -1,  n, alphat, At, lda, Bt, ldb ), blas::Error );
    assert_throw( tmblas::trmm( side,    uplo,    trans, diag,     m, -1, alphat, At, lda, Bt, ldb ), blas::Error );

    assert_throw( tmblas::trmm( Side::Left,  uplo,   trans, diag,     m,  n, alphat, At, m-1, Bt, ldb ), blas::Error );
    assert_throw( tmblas::trmm( Side::Right, uplo,   trans, diag,     m,  n, alphat, At, n-1, Bt, ldb ), blas::Error );

    assert_throw( tmblas::trmm( side, uplo, trans, diag,    m,  n, alphat, At, lda, Bt, m-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, Am=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm=%.2e\n",
                llong( Am ), llong( Am ), llong( lda ), llong( size_A ), Anorm,
                llong( Bm ), llong( Bn ), llong( ldb ), llong( size_B ), Bnorm );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( Am, Am, A, lda );
        printf( "B = " ); print_matrix( Bm, Bn, B, ldb );
    }
#endif
    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::trmm<TA, TB>( side, uplo, trans, diag, m, n, alphat, At, lda, Bt, ldb );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::trmm( side, m, n );
    params.time()   = time;
    params.gflops() = gflop / time;
#if 0
    if (verbose >= 2) {
        printf( "X = " ); print_matrix( Bm, Bn, B, ldb );
    }
#endif
    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
	blas::trmm<RA, RB>( layout, side, uplo, trans, diag,
			    m, n, alphar, Ar, lda, Br, ldb );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;
#if 0
        if (verbose >= 2) {
            printf( "Xref = " ); print_matrix( Bm, Bn, Bref, ldb );
        }
#endif
        // check error compared to reference
	double errormax, errorl2;
        bool okay = true;
	check_diff<RB, TB>(Bm, Bn, Br, Bt, ldb, &errormax, &errorl2);
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
    delete[] Ar;
    delete[] Br;
    delete[] At;
    delete[] Bt;
}

// -----------------------------------------------------------------------------
void test_trmm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
	  test_trmm_work< float, float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
	  test_trmm_work< double, double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
	  test_trmm_work< quadruple, quadruple, quadruple, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::Octuple:
	  test_trmm_work< octuple, octuple, octuple, octuple, octuple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_trmm_work< std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trmm_work< std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
	  test_trmm_work< std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

	    case testsweeper::DataType::OctupleComplex:
	  test_trmm_work< std::complex<octuple>, std::complex<octuple>, std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
