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
template <typename Tbase, typename RX, typename TX>
void test_asum_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using scalar_t = blas::scalar_type< RX >;
    using scalar_tt = blas::scalar_type< TX >;

    bool trunc = true;
    // get & mark input values
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
    RX* xr   = new RX[ size_x ];
    TX* xt   = new TX[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv( idist, iseed, size_x, x, trunc );

#if 0
    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );
#endif

    for (size_t i = 0; i < size_x; i++) {
      xt[i] = tmblas::type_conv<TX, Tbase>(x[i]);
      xr[i] = tmblas::type_conv<RX, Tbase>(x[i]);		
    }
#if 0
    // test error exits
    assert_throw( tmblas::asum( -1, xt, incx), blas::Error );
    assert_throw( tmblas::asum(  n, xt,    0), blas::Error );
    assert_throw( tmblas::asum(  n, xt, incx), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n",
                llong( n ), llong( incx ), llong( size_x ), Xnorm,
                llong( n ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "x = " ); print_vector( n, x, incx );
        printf( "y = " ); print_vector( n, y, incy );
    }
#endif

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    scalar_tt resultt = tmblas::asum <TX> ( n, xt, incx);
    time = get_wtime() - time;

//	scalar_t result = tmblas::type_conv<scalar_t, scalar_tt>(resultt);

    double gflop = blas::Gflop< scalar_t >::asum( n );
    double gbyte = blas::Gbyte< scalar_t >::asum( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

#if 0
    if (verbose >= 1) {
        printf( "asum = %.4e + %.4ei\n", real(result), imag(result) );
    }
#endif

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        scalar_t resultr = blas::asum <RX> ( n, xr, incx);
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

#if 0
        if (verbose >= 1) {
            printf( "ref = %.4e + %.4ei\n", real(ref), imag(ref) );
        }
#endif

        // check error compared to reference
        // treat result as 1 x 1 matrix; k = n is reduction dimension
        // alpha=1, beta=0, Cnorm=0
        double errormax, errorl2;
        bool okay = true;
//        check_gemm( 1, 1, n, scalar_t(1), scalar_t(0), Xnorm, Ynorm, real_t(0),
//                    &ref, 1, &result, 1, verbose, &error, &okay );
		check_diff<scalar_t, scalar_tt>(1, 1, &resultr, &resultt, 1, &errormax, &errorl2);
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
    delete[] xr;
    delete[] xt;
}

// -----------------------------------------------------------------------------
void test_asum( Params& params, bool run )
{
    switch (params.datatype()) {
//        case testsweeper::DataType::Half:
//            test_asum_work< float, float, half, half >( params, run );
//            break;

        case testsweeper::DataType::Single:
            test_asum_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_asum_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_asum_work< double, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::Octuple:
            test_asum_work< double, octuple, octuple >( params, run );
            break;

//        case testsweeper::DataType::HalfComplex:
//            test_asum_work< std::complex<float>, std::complex<float>, std::complex<half>, std::complex<half> >
//                ( params, run );
//            break;

        case testsweeper::DataType::SingleComplex:
            test_asum_work< std::complex<float>, std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_asum_work< std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_asum_work< std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

        case testsweeper::DataType::OctupleComplex:
            test_asum_work< std::complex<double>, std::complex<octuple>, std::complex<octuple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
