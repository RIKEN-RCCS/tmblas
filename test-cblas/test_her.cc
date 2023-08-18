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
void test_her_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Uplo;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RX >;
    using scalar_tt = blas::scalar_type< TA, TX >;
    using real_t   = blas::real_type< scalar_t >;
    using real_tt  = blas::real_type< scalar_tt >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    real_t alpha    = params.alpha();  // note: real
    real_tt alphat  = tmblas::type_conv<real_tt, real_t>(params.alpha());
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
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    RA* A    = new RA[ size_A ];
    RA* Aref = new RA[ size_A ];
    RX* x    = new RX[ size_x ];
    TA* At   = new TA[ size_A ];
    TX* xt   = new TX[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_lacpy( "g", n, n, A, lda, Aref, lda );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lanhe( "f", uplo2str(uplo), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_x; i++)
		xt[i] = tmblas::type_conv<TX, RX>(x[i]);

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::her<TA,TX>( Uplo(0),  n, alphat, xt, incx, At, lda )), blas::Error );
    assert_throw(( tmblas::her<TA,TX>( uplo,    -1, alphat, xt, incx, At, lda )), blas::Error );
    assert_throw(( tmblas::her<TA,TX>( uplo,     n, alphat, xt,    0, At, lda )), blas::Error );
    assert_throw(( tmblas::her<TA,TX>( uplo,     n, alphat, xt, incx, At, n-1 )), blas::Error );
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e;\n", alpha );
        printf( "A = " ); print_matrix( n, n, A, lda );
        printf( "x = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::her<TA,TX>( uplo, n, alphat, xt, incx, At, lda );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_A; i++)
		A[i] = tmblas::type_conv<RA, TA>(At[i]);

    double gflop = blas::Gflop< scalar_t >::her( n );
    double gbyte = blas::Gbyte< scalar_t >::her( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, A, lda );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_her( cblas_layout_const(layout), cblas_uplo_const(uplo),
                   n, alpha, x, incx, Aref, lda );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( n, n, Aref, lda );
        }

        // check error compared to reference
        // beta = 1
        real_t error;
        bool okay;
        check_herk( uplo, n, 1, alpha, real_t(1), Xnorm, Xnorm, Anorm,
                    Aref, lda, A, lda, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] Aref;
    delete[] x;
    delete[] At;
    delete[] xt;
}

// -----------------------------------------------------------------------------
void test_her( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
        case testsweeper::DataType::Double:
        case testsweeper::DataType::Quadruple:

        case testsweeper::DataType::SingleComplex:
            test_her_work< std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_her_work< std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_her_work< std::complex<double>, std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
