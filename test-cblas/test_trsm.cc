//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template <typename RA, typename RB, typename TA, typename TB>
void test_trsm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Uplo;
    using blas::Side;
    using blas::Op;
    using blas::Layout;
    using blas::Diag;
    using scalar_t = blas::scalar_type< RA, RB >;
    using scalar_tt = blas::scalar_type< TA, TB >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    blas::Op trans  = params.trans();
    blas::Diag diag = params.diag();
    scalar_t alpha  = params.alpha();
    scalar_tt alphat = tmblas::type_conv<scalar_tt, scalar_t>(params.alpha());
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

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
    RA* A    = new RA[ size_A ];
    RB* B    = new RB[ size_B ];
    RB* Bref = new RB[ size_B ];
    TA* At   = new TA[ size_A ];
    TB* Bt   = new TB[ size_B ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );  // TODO: generate
    lapack_larnv( idist, iseed, size_B, B );  // TODO
    lapack_lacpy( "g", Bm, Bn, B, ldb, Bref, ldb );

    // set unused data to nan
    if (uplo == Uplo::Lower) {
        for (int64_t j = 0; j < Am; ++j)
            for (int64_t i = 0; i < j; ++i)  // upper
                A[ i + j*lda ] = nan("");
    }
    else {
        for (int64_t j = 0; j < Am; ++j)
            for (int64_t i = j+1; i < Am; ++i)  // lower
                A[ i + j*lda ] = nan("");
    }

    // Factor A into L L^H or U U^H to get a well-conditioned triangular matrix.
    // If diag == Unit, the diagonal is replaced; this is still well-conditioned.
    // First, brute force positive definiteness.
    for (int64_t i = 0; i < Am; ++i) {
        A[ i + i*lda ] += Am;
    }
    int64_t info = 0;
    lapack_potrf( uplo2str(uplo), Am, A, lda, &info );
    require( info == 0 );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lantr( "f", uplo2str(uplo), diag2str(diag),
                                 Am, Am, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );

    // if row-major, transpose A
    if (layout == Layout::RowMajor) {
        for (int64_t j = 0; j < Am; ++j) {
            for (int64_t i = 0; i < j; ++i) {
                std::swap( A[ i + j*lda ], A[ j + i*lda ] );
            }
        }
    }

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_B; i++)
		Bt[i] = tmblas::type_conv<TB, RB>(B[i]);

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::trsm<TA,TB>( Side(0), uplo,    trans, diag,     m,  n, alphat, At, lda, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::trsm<TA,TB>( side,    Uplo(0), trans, diag,     m,  n, alphat, At, lda, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::trsm<TA,TB>( side,    uplo,    Op(0), diag,     m,  n, alphat, At, lda, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::trsm<TA,TB>( side,    uplo,    trans, Diag(0),  m,  n, alphat, At, lda, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::trsm<TA,TB>( side,    uplo,    trans, diag,    -1,  n, alphat, At, lda, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::trsm<TA,TB>( side,    uplo,    trans, diag,     m, -1, alphat, At, lda, Bt, ldb )), blas::Error );

    assert_throw(( tmblas::trsm<TA,TB>( Side::Left,  uplo,   trans, diag,     m,  n, alphat, At, m-1, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::trsm<TA,TB>( Side::Right, uplo,   trans, diag,     m,  n, alphat, At, n-1, Bt, ldb )), blas::Error );

    assert_throw(( tmblas::trsm<TA,TB>( side, uplo, trans, diag,    m,  n, alphat, At, lda, Bt, m-1 )), blas::Error );
	#endif

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

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::trsm<TA,TB>( side, uplo, trans, diag, m, n, alphat, At, lda, Bt, ldb );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_B; i++)
		B[i] = tmblas::type_conv<RB, TB>(Bt[i]);

    double gflop = blas::Gflop< scalar_t >::trsm( side, m, n );
    params.time()   = time;
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "X = " ); print_matrix( Bm, Bn, B, ldb );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_trsm( cblas_layout_const(layout),
                    cblas_side_const(side),
                    cblas_uplo_const(uplo),
                    cblas_trans_const(trans),
                    cblas_diag_const(diag),
                    m, n, alpha, A, lda, Bref, ldb );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Xref = " ); print_matrix( Bm, Bn, Bref, ldb );
        }

        // check error compared to reference
        // Am is reduction dimension
        // beta = 0, Cnorm = 0 (initial).
        real_t error;
        bool okay;
        check_gemm( Bm, Bn, Am, alpha, scalar_t(0), Anorm, Bnorm, real_t(0),
                    Bref, ldb, B, ldb, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] Bref;
    delete[] At;
    delete[] Bt;
}

// -----------------------------------------------------------------------------
void test_trsm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trsm_work< float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_trsm_work< double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_trsm_work< double, double, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_trsm_work< std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trsm_work< std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_trsm_work< std::complex<double>, std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
