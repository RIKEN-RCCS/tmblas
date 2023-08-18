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
template <typename Tbase, typename RA, typename RX, typename TA, typename TX>
void test_syr_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RX >;
    using scalar_tt = blas::scalar_type< TA, TX >;

    bool trunc = true;
    
    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo ();
    Tbase alpha;
    tmblas_setscal<Tbase>(alpha, params.alpha(), params.alphai());
    if (trunc) {
      alpha = tmblas_trunc<Tbase>(alpha);
    }
    scalar_t alphar = tmblas::type_conv<scalar_t,  Tbase>(alpha);
    scalar_tt alphat = tmblas::type_conv<scalar_tt, Tbase>(alpha);
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
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
    Tbase* A = new Tbase[ size_A ];
    Tbase* x = new Tbase[ size_x ];
    RA* Ar   = new RA[ size_A ];
    RX* xr   = new RX[ size_x ];
    TA* At   = new TA[ size_A ];
    TX* xt   = new TX[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv( idist, iseed, size_A, A, trunc );
    tmblas_larnv( idist, iseed, size_x, x, trunc );
    //lapack_lacpy( "g", n, n, A, lda, Aref, lda );

#if 0
    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lansy( "f", uplo2str(uplo), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
#endif

	for (size_t i = 0; i < size_A; i++) {
		At[i] = tmblas::type_conv<TA, Tbase>(A[i]);
		Ar[i] = tmblas::type_conv<RA, Tbase>(A[i]);
	}
	for (size_t i = 0; i < size_x; i++) {
		xt[i] = tmblas::type_conv<TX, Tbase>(x[i]);
		xr[i] = tmblas::type_conv<RX, Tbase>(x[i]);
	}

#if 0
    // test error exits
    assert_throw( tmblas::syr( Uplo(0),  n, alphat, xt, incx, At, lda ), blas::Error );
    assert_throw( tmblas::syr( uplo,    -1, alphat, xt, incx, At, lda ), blas::Error );
    assert_throw( tmblas::syr( uplo,     n, alphat, xt,    0, At, lda ), blas::Error );
    assert_throw( tmblas::syr( uplo,     n, alphat, xt, incx, At, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "A = " ); print_matrix( n, n, A, lda );
        printf( "x = " ); print_vector( n, x, incx );
    }
#endif

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::syr <TX, TA> ( uplo, n, alphat, xt, incx, At, lda );
    time = get_wtime() - time;

//	for (size_t i = 0; i < size_A; i++)
//		A[i] = tmblas::type_conv<RA, TA>(At[i]);

    double gflop = blas::Gflop< scalar_t >::syr( n );
    double gbyte = blas::Gbyte< scalar_t >::syr( n );
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
        blas::syr <RX, RA> ( layout, uplo,
                   n, alphar, xr, incx, Ar, lda );
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
//        check_herk( uplo, n, 1, alpha, scalar_t(1), Xnorm, Xnorm, Anorm,
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
    delete[] Ar;
    delete[] xr;
    delete[] At;
    delete[] xt;
}

// -----------------------------------------------------------------------------
void test_syr( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_syr_work< float,
			   float, float,
			   float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_syr_work< double,
			   double, double,
			   double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_syr_work< double,
			   quadruple, quadruple,
			   quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::Octuple:
            test_syr_work< double,
			   octuple, octuple,
			   octuple, octuple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_syr_work< std::complex<float>,
			   std::complex<float>, std::complex<float>,
			   std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_syr_work< std::complex<double>,
			   std::complex<double>, std::complex<double>,
			   std::complex<double>, std::complex<double> >( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_syr_work< std::complex<double>, 
			   std::complex<quadruple>, std::complex<quadruple>,
			   std::complex<quadruple>, std::complex<quadruple> >( params, run );
            break;

        case testsweeper::DataType::OctupleComplex:
            test_syr_work< std::complex<double>, 
			   std::complex<octuple>, std::complex<octuple>,
			   std::complex<octuple>, std::complex<octuple> >( params, run );
            break;
        default:
            throw std::exception();
            break;
    }
}
