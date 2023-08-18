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
template <typename RA, typename RX, typename RY, typename TA, typename TX, typename TY>
void test_symv_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RX, RY >;
    using scalar_tt = blas::scalar_type< TA, TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    scalar_tt alphat = tmblas::type_conv<scalar_tt, scalar_t>(params.alpha());
    scalar_tt betat  = tmblas::type_conv<scalar_tt, scalar_t>(params.beta());
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
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
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    RA* A    = new RA[ size_A ];
    RX* x    = new RX[ size_x ];
    RY* y    = new RY[ size_y ];
    RY* yref = new RY[ size_y ];
    TA* At   = new TA[ size_A ];
    TX* xt   = new TX[ size_x ];
    TY* yt   = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, y, incy, yref, incy );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lansy( "f", uplo2str(uplo), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_x; i++)
		xt[i] = tmblas::type_conv<TX, RX>(x[i]);
	for (size_t i = 0; i < size_y; i++)
		yt[i] = tmblas::type_conv<TY, RY>(y[i]);

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::symv<TA,TX,TY>( Uplo(0),  n, alphat, At, lda, xt, incx, betat, yt, incy )), blas::Error );
    assert_throw(( tmblas::symv<TA,TX,TY>( uplo,    -1, alphat, At, lda, xt, incx, betat, yt, incy )), blas::Error );
    assert_throw(( tmblas::symv<TA,TX,TY>( uplo,     n, alphat, At, n-1, xt, incx, betat, yt, incy )), blas::Error );
    assert_throw(( tmblas::symv<TA,TX,TY>( uplo,     n, alphat, At, lda, xt,    0, betat, yt, incy )), blas::Error );
    assert_throw(( tmblas::symv<TA,TX,TY>( uplo,     n, alphat, At, lda, xt, incx, betat, yt,    0 )), blas::Error );
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm,
                llong( n ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( n, n, A, lda );
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::symv<TA,TX,TY>( uplo, n, alphat, At, lda, xt, incx, betat, yt, incy );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_y; i++)
		y[i] = tmblas::type_conv<RY, TY>(yt[i]);

    double gflop = blas::Gflop< scalar_t >::symv( n );
    double gbyte = blas::Gbyte< scalar_t >::symv( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_symv( cblas_layout_const(layout), cblas_uplo_const(uplo), n,
                    alpha, A, lda, x, incx, beta, yref, incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // check error compared to reference
        // treat y as 1 x leny matrix with ld = incy; k = lenx is reduction dimension
        real_t error;
        bool okay;
        check_gemm( 1, n, n, alpha, beta, Anorm, Xnorm, Ynorm,
                    yref, std::abs(incy), y, std::abs(incy), verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] x;
    delete[] y;
    delete[] yref;
    delete[] At;
    delete[] xt;
    delete[] yt;
}

// -----------------------------------------------------------------------------
void test_symv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_symv_work< float, float, float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_symv_work< double, double, double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_symv_work< double, double, double, quadruple, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
        case testsweeper::DataType::DoubleComplex:
        case testsweeper::DataType::QuadrupleComplex:
            throw blas::Error( "See symv< complex > in LAPACK++", __func__ );
            break;

        default:
            throw std::exception();
            break;
    }
}
