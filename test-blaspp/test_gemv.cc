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
void test_gemv_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RX, RY >;
    using scalar_tt = blas::scalar_type< TA, TX, TY >;

    bool trunc = true;
    
    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();

    Tbase alpha, beta;
    tmblas_setscal<Tbase>(alpha, params.alpha(), params.alphai());
    tmblas_setscal<Tbase>(beta,  params.beta(),  params.betai());
    if (trunc) {
      alpha = tmblas_trunc<Tbase>(alpha);
      beta = tmblas_trunc<Tbase>(beta);
    }
    scalar_t alphar = tmblas::type_conv<scalar_t,  Tbase>(alpha);
    scalar_tt alphat = tmblas::type_conv<scalar_tt, Tbase>(alpha);
    scalar_t betar = tmblas::type_conv<scalar_t,  Tbase>(beta);
    scalar_tt betat = tmblas::type_conv<scalar_tt, Tbase>(beta);

    int64_t m       = params.dim.m();
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
    int64_t Am = (layout == Layout::ColMajor ? m : n);
    int64_t An = (layout == Layout::ColMajor ? n : m);
    int64_t lda = roundup( Am, align );
    int64_t Xm = (trans == Op::NoTrans ? n : m);
    int64_t Ym = (trans == Op::NoTrans ? m : n);
    size_t size_A = size_t(lda)*An;
    size_t size_x = (Xm - 1) * std::abs(incx) + 1;
    size_t size_y = (Ym - 1) * std::abs(incy) + 1;
    Tbase* A    = new Tbase[ size_A ];
    Tbase* x    = new Tbase[ size_x ];
    Tbase* y    = new Tbase[ size_y ];
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
    //cblas_copy( Ym, y, incy, yref, incy );

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
    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Xnorm = cblas_nrm2( Xm, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( Ym, y, std::abs(incy) );

    // test error exits
    assert_throw( tmblas::gemv( Op(0),  m,  n, alphat, At, lda, xt, incx, betat, yt, incy ), blas::Error );
    assert_throw( tmblas::gemv( trans, -1,  n, alphat, At, lda, xt, incx, betat, yt, incy ), blas::Error );
    assert_throw( tmblas::gemv( trans,  m, -1, alphat, At, lda, xt, incx, betat, yt, incy ), blas::Error );

    assert_throw( tmblas::gemv( trans,  m,  n, alphat, At, m-1, xt, incx, betat, yt, incy ), blas::Error );

    assert_throw( tmblas::gemv( trans,  m,  n, alphat, At, lda, xt, 0,     betat, yt, incy ), blas::Error );
    assert_throw( tmblas::gemv( trans,  m,  n, alphat, At, lda, xt, incx, betat, yt, 0    ), blas::Error );

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
#endif

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::gemv <TA, TX, TY> ( trans, m, n, alphat, At, lda, xt, incx, betat, yt, incy );
    time = get_wtime() - time;

//	for (size_t i = 0; i < size_y; i++)
//		y[i] = tmblas::type_conv<RY, TY>(yt[i]);

    double gflop = blas::Gflop< scalar_t >::gemv( m, n );
    double gbyte = blas::Gbyte< scalar_t >::gemv( m, n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

#if 0
    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( Ym, y, incy );
    }
#endif

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        blas::gemv <RA, RX, RY> ( layout, trans, m, n,
                    alphar, Ar, lda, xr, incx, betar, yr, incy );
        time = get_wtime() - time;

        params.ref_time()   = time; // * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

#if 0
        if (verbose >= 2) {
            printf( "yref = " ); print_vector( Ym, yref, incy );
        }
#endif

        // check error compared to reference
        // treat y as 1 x Ym matrix with ld = incy; k = Xm is reduction dimension
        double errormax, errorl2;
        bool okay = true;
//        check_gemm( 1, Ym, Xm, alpha, beta, Anorm, Xnorm, Ynorm,
//                    yref, std::abs(incy), y, std::abs(incy), verbose, &error, &okay );
	check_diff<RY, TY>(Ym, 1, yr, yt, Ym, &errormax, &errorl2);
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
void test_gemv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gemv_work< float,
			    float, float, float,
			    float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gemv_work< double,
			    double, double, double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_gemv_work< double,
			    quadruple, quadruple, quadruple,
			    quadruple, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::Octuple:
            test_gemv_work< double,
			    octuple, octuple, octuple,
			    octuple, octuple, octuple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemv_work< std::complex<float>,
			    std::complex<float>, std::complex<float>, std::complex<float>,
			    std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

    case testsweeper::DataType::DoubleComplex:
      test_gemv_work< std::complex<double>,
		      std::complex<double>, std::complex<double>, std::complex<double>,
		      std::complex<double>, std::complex<double>, std::complex<double>>( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
	  test_gemv_work< std::complex<double>,
			  std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>,
			  std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>>( params, run );
            break;

        case testsweeper::DataType::OctupleComplex:
	  test_gemv_work< std::complex<double>,
			  std::complex<octuple>, std::complex<octuple>, std::complex<octuple>,
			  std::complex<octuple>, std::complex<octuple>, std::complex<octuple>>( params, run );
	  break;

        default:
            throw std::exception();
            break;
    }
}
