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
template <typename Tbase, typename RX, typename RY, typename TX, typename TY>
void test_axpy_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using scalar_t = blas::scalar_type< RX, RY >;
    using scalar_tt = blas::scalar_type< TX, TY >;

    bool trunc = true;

    Tbase alpha;
    tmblas_setscal<Tbase>(alpha, params.alpha(), params.alphai());
    if (trunc) {
      alpha = tmblas_trunc<Tbase>(alpha);
    }
    scalar_t alphar = tmblas::type_conv<scalar_t,  Tbase>(alpha);
    scalar_tt alphat = tmblas::type_conv<scalar_tt, Tbase>(alpha);

    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
//    int64_t verbose = params.verbose();

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
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    Tbase* x = new Tbase[ size_x ];
    Tbase* y = new Tbase[ size_y ];
    TX* xt   = new TX[ size_x ];
    TY* yt   = new TY[ size_y ];
    RY* xr   = new RX[ size_x ];
    RY* yr   = new RY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv( idist, iseed, size_x, x, trunc );
    tmblas_larnv( idist, iseed, size_y, y, trunc );
//    cblas_copy( n, y, incy, yref, incy );
//    cblas_copy( n, y, incy, y0,   incy );

	for (size_t i = 0; i < size_x; i++) {
		xt[i] = tmblas::type_conv<TX, Tbase>(x[i]);
		xr[i] = tmblas::type_conv<RX, Tbase>(x[i]);
	}
	for (size_t i = 0; i < size_y; i++) {
		yt[i] = tmblas::type_conv<TY, Tbase>(y[i]);
		yr[i] = tmblas::type_conv<RY, Tbase>(y[i]);
	}

#if 0
    // test error exits
    assert_throw( tmblas::axpy( -1, alphat, xt, incx, yt, incy ), blas::Error );
    assert_throw( tmblas::axpy(  n, alphat, xt,    0, yt, incy ), blas::Error );
    assert_throw( tmblas::axpy(  n, alphat, xt, incx, yt,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ),
                llong( n ), llong( incy ), llong( size_y ) );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }
#endif

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::axpy <TX, TY>( n, alphat, xt, incx, yt, incy );
    time = get_wtime() - time;

//	for (size_t i = 0; i < size_y; i++)
//		y[i] = tmblas::type_conv<RY, TY>(yt[i]);

    double gflop = blas::Gflop< scalar_t >::axpy( n );
    double gbyte = blas::Gbyte< scalar_t >::axpy( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

#if 0
    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, y, incy );
    }
#endif

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        blas::axpy <RX, RY>( n, alphar, xr, incx, yr, incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

#if 0
        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // maximum component-wise forward error:
        // | fl(yi) - yi | / (2 |alpha xi| + |y0_i|)
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] = std::abs( y[iy] - yref[iy] )
                  / (2*(std::abs( alpha * x[ix] ) + std::abs( y0[iy] )));
            error = std::max( error, real( y[iy] ) );
            ix += incx;
            iy += incy;
        }

        if (verbose >= 2) {
            printf( "err  = " ); print_vector( n, y, incy, "%9.2e" );
        }

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex<scalar_t>::value) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
#endif
        double errormax, errorl2;
        bool okay = true;
	check_diff<RY, TY>(n, 1, yr, yt, n, &errormax, &errorl2);
	if (trunc) {
	  if ((errormax != 0.0) || (errorl2 != 0.0)){
	    okay = false;
	  }
	}
        params.error() = errormax;
        params.error2() = errorl2;
        params.okay() = okay;
    }

    delete[] x;
    delete[] y;
    delete[] xr;
    delete[] yr;
    delete[] xt;
    delete[] yt;
}

// -----------------------------------------------------------------------------
void test_axpy( Params& params, bool run )
{
    switch (params.datatype()) {
//        case testsweeper::DataType::Half:
//            test_axpy_work< float, float, half, half >( params, run );
//            break;

        case testsweeper::DataType::Single:
            test_axpy_work< float, float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_axpy_work< double, double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_axpy_work< double, quadruple, quadruple, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::Octuple:
            test_axpy_work< double, octuple, octuple, octuple, octuple >( params, run );
            break;

//        case testsweeper::DataType::HalfComplex:
//            test_axpy_work< std::complex<float>, std::complex<float>, std::complex<half>, std::complex<half> >
//                ( params, run );
//            break;

        case testsweeper::DataType::SingleComplex:
            test_axpy_work< std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_axpy_work< std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_axpy_work< std::complex<double>, std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

        case testsweeper::DataType::OctupleComplex:
            test_axpy_work< std::complex<double>, std::complex<octuple>, std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
