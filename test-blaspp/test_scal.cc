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
template <typename Tbase, typename R, typename T>
void test_scal_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;

    bool trunc = true;

    Tbase alpha;
    tmblas_setscal<Tbase>(alpha, params.alpha(), params.alphai());
    if (trunc) {
      alpha = tmblas_trunc<Tbase>(alpha);
    }
    R alphar = tmblas::type_conv<R,  Tbase>(alpha);
    T alphat = tmblas::type_conv<T, Tbase>(alpha);    

    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
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
    Tbase* x = new Tbase[ size_x ];
    R* xr    = new R[ size_x ];
    T* xt    = new T[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv( idist, iseed, size_x, x, trunc );
    //cblas_copy( n, x, incx, xref, incx );

	for (size_t i = 0; i < size_x; i++) {
		xt[i] = tmblas::type_conv<T, Tbase>(x[i]);
		xr[i] = tmblas::type_conv<R, Tbase>(x[i]);
	}

#if 0
    // test error exits
    assert_throw( tmblas::scal( -1, alphat, xt, incx ), blas::Error );
    assert_throw( tmblas::scal(  n, alphat, xt,    0 ), blas::Error );
    assert_throw( tmblas::scal(  n, alphat, xt,   -1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
    }
#endif

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::scal <T> ( n, alphat, xt, incx );
    time = get_wtime() - time;

//	for (size_t i = 0; i < size_x; i++)
//		x[i] = tmblas::type_conv<R, T>(xt[i]);

    double gflop = blas::Gflop< R >::scal( n );
    double gbyte = blas::Gbyte< R >::scal( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

#if 0
    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }
#endif

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        blas::scal <R> ( n, alphar, xr, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

#if 0
        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
        }

        // maximum component-wise forward error:
        // | fl(xi) - xi | / | xi |
        real_t error = 0;
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            error = std::max( error, std::abs( (xref[ix] - x[ix]) / xref[ix] ));
            ix += incx;
        }
        params.error() = error;

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex<R>::value) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
#endif
        double errormax, errorl2;
        bool okay = true;
	check_diff<R, T>(n, 1, xr, xt, n, &errormax, &errorl2);
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
    delete[] xt;
    delete[] xr;
}

// -----------------------------------------------------------------------------
void test_scal( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_scal_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_scal_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_scal_work< double, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::Octuple:
            test_scal_work< double, octuple, octuple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_scal_work< std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_scal_work< std::complex<double>, std::complex<double>, std::complex<double> >( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_scal_work< std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >( params, run );
            break;

        case testsweeper::DataType::OctupleComplex:
            test_scal_work< std::complex<double>, std::complex<octuple>, std::complex<octuple> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
