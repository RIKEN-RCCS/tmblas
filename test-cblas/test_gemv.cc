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
void test_gemv_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RX, RY >;
    using scalar_tt = blas::scalar_type< TA, TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    scalar_tt alphat = tmblas::type_conv<scalar_tt, scalar_t>(params.alpha());
    scalar_tt betat  = tmblas::type_conv<scalar_tt, scalar_t>(params.beta());
    int64_t m       = params.dim.m();
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
    int64_t Am = (layout == Layout::ColMajor ? m : n);
    int64_t An = (layout == Layout::ColMajor ? n : m);
    int64_t lda = roundup( Am, align );
    int64_t Xm = (trans == Op::NoTrans ? n : m);
    int64_t Ym = (trans == Op::NoTrans ? m : n);
    size_t size_A = size_t(lda)*An;
    size_t size_x = (Xm - 1) * std::abs(incx) + 1;
    size_t size_y = (Ym - 1) * std::abs(incy) + 1;
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
    cblas_copy( Ym, y, incy, yref, incy );

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_x; i++)
		xt[i] = tmblas::type_conv<TX, RX>(x[i]);
	for (size_t i = 0; i < size_y; i++)
		yt[i] = tmblas::type_conv<TY, RY>(y[i]);

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Xnorm = cblas_nrm2( Xm, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( Ym, y, std::abs(incy) );

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::gemv<TA,TX,TY>( Op(0),  m,  n, alphat, At, lda, xt, incx, betat, yt, incy )), blas::Error );
    assert_throw(( tmblas::gemv<TA,TX,TY>( trans, -1,  n, alphat, At, lda, xt, incx, betat, yt, incy )), blas::Error );
    assert_throw(( tmblas::gemv<TA,TX,TY>( trans,  m, -1, alphat, At, lda, xt, incx, betat, yt, incy )), blas::Error );

    assert_throw(( tmblas::gemv<TA,TX,TY>( trans,  m,  n, alphat, At, m-1, xt, incx, betat, yt, incy )), blas::Error );

    assert_throw(( tmblas::gemv<TA,TX,TY>( trans,  m,  n, alphat, At, lda, xt, 0,     betat, yt, incy )), blas::Error );
    assert_throw(( tmblas::gemv<TA,TX,TY>( trans,  m,  n, alphat, At, lda, xt, incx, betat, yt, 0    )), blas::Error );
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x Xm=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n"
                "y Ym=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n",
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Xm ), llong( incx ), llong( size_x ), Xnorm,
                llong( Ym ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( m, n, A, lda );
        printf( "x    = " ); print_vector( Xm, x, incx );
        printf( "y    = " ); print_vector( Ym, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::gemv<TA,TX,TY>( trans, m, n, alphat, At, lda, xt, incx, betat, yt, incy );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_y; i++)
		y[i] = tmblas::type_conv<RY, TY>(yt[i]);

    double gflop = blas::Gflop< scalar_t >::gemv( m, n );
    double gbyte = blas::Gbyte< scalar_t >::gemv( m, n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( Ym, y, incy );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_gemv( cblas_layout_const(layout), cblas_trans_const(trans), m, n,
                    alpha, A, lda, x, incx, beta, yref, incy );
        time = get_wtime() - time;

        params.ref_time()   = time; // * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( Ym, yref, incy );
        }

        // check error compared to reference
        // treat y as 1 x Ym matrix with ld = incy; k = Xm is reduction dimension
        real_t error;
        bool okay;
        check_gemm( 1, Ym, Xm, alpha, beta, Anorm, Xnorm, Ynorm,
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
void test_gemv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gemv_work< float, float, float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gemv_work< double, double, double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_gemv_work< double, double, double, quadruple, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemv_work< std::complex<float>, std::complex<float>, std::complex<float>,
							std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemv_work< std::complex<double>, std::complex<double>, std::complex<double>,
							std::complex<double>, std::complex<double>, std::complex<double>>( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_gemv_work< std::complex<double>, std::complex<double>, std::complex<double>,
							std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>>( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
