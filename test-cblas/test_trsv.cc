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
template <typename RA, typename RX, typename TA, typename TX>
void test_trsv_work( Params& params, bool run )
{
    #define A(i_, j_) (A + (i_) + (j_)*lda)

    using namespace testsweeper;
    using blas::Uplo;
    using blas::Op;
    using blas::Layout;
    using blas::Diag;
    using scalar_t = blas::scalar_type< RA, RX >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    blas::Op trans  = params.trans();
    blas::Diag diag = params.diag();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    // setup
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = size_t(n - 1) * std::abs(incx) + 1;
    RA* A    = new RA[ size_A ];
    RX* x    = new RX[ size_x ];
    RX* xref = new RX[ size_x ];
    TA* At   = new TA[ size_A ];
    TX* xt   = new TX[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    // set unused data to nan
    if (uplo == Uplo::Lower) {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < j; ++i)  // upper
                A[ i + j*lda ] = nan("");
    }
    else {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j+1; i < n; ++i)  // lower
                A[ i + j*lda ] = nan("");
    }

    // Factor A into L L^H or U U^H to get a well-conditioned triangular matrix.
    // If diag == Unit, the diagonal is replaced; this is still well-conditioned.
    // First, brute force positive definiteness.
    for (int64_t i = 0; i < n; ++i) {
        A[ i + i*lda ] += n;
    }
    int64_t info = 0;
    lapack_potrf( uplo2str(uplo), n, A, lda, &info );
    require( info == 0 );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lantr( "f", uplo2str(uplo), diag2str(diag),
                                 n, n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

    // if row-major, transpose A
    if (layout == Layout::RowMajor) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < j; ++i) {
                std::swap( A[ i + j*lda ], A[ j + i*lda ] );
            }
        }
    }

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_x; i++)
		xt[i] = tmblas::type_conv<TX, RX>(x[i]);

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::trsv<TA,TX>( Uplo(0), trans, diag,     n, At, lda, xt, incx )), blas::Error );
    assert_throw(( tmblas::trsv<TA,TX>( uplo,    Op(0), diag,     n, At, lda, xt, incx )), blas::Error );
    assert_throw(( tmblas::trsv<TA,TX>( uplo,    trans, Diag(0),  n, At, lda, xt, incx )), blas::Error );
    assert_throw(( tmblas::trsv<TA,TX>( uplo,    trans, diag,    -1, At, lda, xt, incx )), blas::Error );
    assert_throw(( tmblas::trsv<TA,TX>( uplo,    trans, diag,     n, At, n-1, xt, incx )), blas::Error );
    assert_throw(( tmblas::trsv<TA,TX>( uplo,    trans, diag,     n, At, lda, xt,    0 )), blas::Error );
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm );
    }
    if (verbose >= 2) {
        printf( "A = "    ); print_matrix( n, n, A, lda );
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::trsv<TA,TX>( uplo, trans, diag, n, At, lda, xt, incx );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_x; i++)
		x[i] = tmblas::type_conv<RX, TX>(xt[i]);

    double gflop = blas::Gflop< scalar_t >::trsv( n );
    double gbyte = blas::Gbyte< scalar_t >::trsv( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        //blas::trsv<RA,RX>( layout,
        cblas_trsv( cblas_layout_const(layout),
                    cblas_uplo_const(uplo),
                    cblas_trans_const(trans),
                    cblas_diag_const(diag),
                    n, A, lda, xref, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
        }

        // check error compared to reference
        // treat x as 1 x n matrix with ld = incx; k = n is reduction dimension
        // alpha = 1, beta = 0.
        real_t error;
        bool okay;
        check_gemm( 1, n, n, scalar_t(1), scalar_t(0), Anorm, Xnorm, real_t(0),
                    xref, std::abs(incx), x, std::abs(incx), verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] x;
    delete[] xref;
    delete[] At;
    delete[] xt;

    #undef A
}

// -----------------------------------------------------------------------------
void test_trsv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trsv_work< float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_trsv_work< double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_trsv_work< double, double, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_trsv_work< std::complex<float>, std::complex<float>,  std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trsv_work< std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_trsv_work< std::complex<double>, std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
