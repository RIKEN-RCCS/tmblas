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
template <typename Tbase, typename RA, typename RX, typename RY, typename TA, typename TX, typename TY>
void test_her2_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RX, RY >;
    using scalar_tt = blas::scalar_type< TA, TX, TY >;
    using real_t   = blas::real_type< scalar_t >;
    using real_tt   = blas::real_type< scalar_tt >;    
    using Tbasereal   = blas::real_type< Tbase >;

    bool trunc = true;
    
    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();

    Tbasereal alpha;
    tmblas_setscal<Tbasereal>(alpha, params.alpha());
    if (trunc) {
      alpha = tmblas_trunc<Tbasereal>(alpha);
    }
    real_t alphar = tmblas::type_conv<real_t,  Tbasereal>(alpha);
    real_tt alphat = tmblas::type_conv<real_tt, Tbasereal>(alpha);
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t align   = params.align();
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
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    Tbase* A = new Tbase[ size_A ];
    Tbase* x = new Tbase[ size_x ];
    Tbase* y = new Tbase[ size_y ];
    RA* Ar   = new RA[ size_A ];
    RX* xr   = new RX[ size_x ];
    RY* yr   = new RY[ size_y ];
    TA* At   = new TA[ size_A ];
    TX* xt   = new TX[ size_x ];
    TY* yt   = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv( idist, iseed, size_A, A, trunc);
    tmblas_larnv( idist, iseed, size_x, x, trunc);
    tmblas_larnv( idist, iseed, size_y, y, trunc);
    //lapack_lacpy( "g", n, n, A, lda, Aref, lda );

#if 0
    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lanhe( "f", uplo2str(uplo), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );
#endif

	for (size_t i = 0; i < size_A; i++) {
		At[i] = tmblas::type_conv<TA, Tbase>(A[i]);
		Ar[i] = tmblas::type_conv<RA, Tbase>(A[i]);
	}
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
    assert_throw( tmblas::her2( Uplo(0),  n, alphat, xt, incx, yt, incy, At, lda ), blas::Error );
    assert_throw( tmblas::her2( uplo,    -1, alphat, xt, incx, yt, incy, At, lda ), blas::Error );
    assert_throw( tmblas::her2( uplo,     n, alphat, xt,    0, yt, incy, At, lda ), blas::Error );
    assert_throw( tmblas::her2( uplo,     n, alphat, xt, incx, yt,    0, At, lda ), blas::Error );
    assert_throw( tmblas::her2( uplo,     n, alphat, xt, incx, yt, incy, At, n-1 ), blas::Error );

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
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "A = " ); print_matrix( n, n, A, lda );
        printf( "x = " ); print_vector( n, x, incx );
        printf( "y = " ); print_vector( n, y, incy );
    }
#endif

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::her2 <TX, TY, TA> ( uplo, n, alphat, xt, incx, yt, incy, At, lda );
    time = get_wtime() - time;

//	for (size_t i = 0; i < size_A; i++)
//		A[i] = tmblas::type_conv<RA, TA>(At[i]);

    double gflop = blas::Gflop< scalar_t >::her2( n );
    double gbyte = blas::Gbyte< scalar_t >::her2( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

#if 0
    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, A, lda );
    }
#endif

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        blas::her2 <RX, RY, RA> ( layout, uplo,
                    n, alphar, xr, incx, yr, incy, Ar, lda );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

#if 0
        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( n, n, Aref, lda );
        }
#endif

        // check error compared to reference
        // beta = 1
        double errormax, errorl2;
        bool okay = true;
//        check_herk( uplo, n, 2, alpha, real_t(1), Xnorm, Ynorm, Anorm,
//                    Aref, lda, A, lda, verbose, &error, &okay );
	check_diff<RA, TA>(n, n, Ar, At, lda, &errormax, &errorl2, uplo);
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
    delete[] x;
    delete[] y;
    delete[] Ar;
    delete[] xr;
    delete[] yr;
    delete[] At;
    delete[] xt;
    delete[] yt;
}

// -----------------------------------------------------------------------------
void test_her2( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
        case testsweeper::DataType::Double:
        case testsweeper::DataType::Quadruple:
        case testsweeper::DataType::Octuple:

        case testsweeper::DataType::SingleComplex:
            test_her2_work< std::complex<float>,
			    std::complex<float>, std::complex<float>, std::complex<float>,
			    std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_her2_work< std::complex<double>,
			    std::complex<double>, std::complex<double>, std::complex<double>,
			    std::complex<double>, std::complex<double>, std::complex<double> >( params, run );
            break;

    case testsweeper::DataType::QuadrupleComplex:
      test_her2_work< std::complex<double>,
		      std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>,
		      std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >( params, run );
      break;
      
    case testsweeper::DataType::OctupleComplex:
      test_her2_work< std::complex<double>,
		      std::complex<octuple>, std::complex<octuple>, std::complex<octuple>,
		      std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >( params, run );
      break;

    default:
      throw std::exception();
      break;
    }
}
