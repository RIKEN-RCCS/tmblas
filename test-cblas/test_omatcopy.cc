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

template <typename T>
class Gbyte_
{
public:
    // read A; write B
    static double omatcopy( double m, double n )
        { return 1e-9 * (2*m*n * sizeof(T)); }
};

inline double fmuls_omatcopy( double m, double n )
    { return m*n; }

template <typename T>
class Gflop_
{
public:
    static constexpr double mul_ops = blas::FlopTraits<T>::mul_ops;

    static double omatcopy( double m, double n )
        { return 1e-9 * (mul_ops*fmuls_omatcopy(m, n)); }
};


// -----------------------------------------------------------------------------
template <typename RA, typename RB, typename TA, typename TB>
void test_omatcopy_work( Params& params, bool run )
{
	#if !defined (INTEL_MKL)
	printf ("Omatcopy test requires Intel MKL.\n");
	exit (1);
	#endif
    using namespace testsweeper;
    using scalar_t = blas::scalar_type< RA, RB >;
    using scalar_tt = blas::scalar_type< TA, TB >;
    using real_t   = blas::real_type< scalar_t >;
    using blas::Op;
    using blas::Layout;
    using std::real;
    using std::imag;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    scalar_t alpha  = params.alpha();
    scalar_tt alphat = tmblas::type_conv<scalar_tt, scalar_t>(params.alpha());
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
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
    int64_t Am = m;
    int64_t An = n;
    int64_t Bm = ((trans == Op::NoTrans || blas::op2char(trans) == 'R' || blas::op2char(trans) == 'r') ? m : n);
    int64_t Bn = ((trans == Op::NoTrans || blas::op2char(trans) == 'R' || blas::op2char(trans) == 'r') ? n : m);
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
    }
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    RA* A    = new RA[ size_A ];
    RB* B    = new RB[ size_B ];
    RB* Bref = new RB[ size_B ];
    TA* At   = new TA[ size_A ];
    TB* Bt   = new TB[ size_B ];
	
	for (size_t i = 0; i < size_B; i++) 
		Bref[i] = B[i] = 0.;
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_lacpy( "g", Bm, Bn, B, ldb, Bref, ldb );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_B; i++)
		Bt[i] = tmblas::type_conv<TB, RB>(B[i]);

    // test error exits
	#if !defined (INTEL_MKL)
	assert_throw(( tmblas::omatcopy<TA,TB>( Op(0),  m,  n,  alphat, At, lda, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::omatcopy<TA,TB>( trans, -1,  n,  alphat, At, lda, Bt, ldb )), blas::Error );
    assert_throw(( tmblas::omatcopy<TA,TB>( trans,  m, -1,  alphat, At, lda, Bt, ldb )), blas::Error );

    assert_throw(( tmblas::omatcopy<TA,TB>( Op::NoTrans,   m, n, alphat, At, lda, Bt, m-1 )), blas::Error );
    assert_throw(( tmblas::omatcopy<TA,TB>( Op::Trans,     m, n, alphat, At, lda, Bt, n-1 )), blas::Error );
    assert_throw(( tmblas::omatcopy<TA,TB>( Op::ConjTrans, m, n, alphat, At, lda, Bt, n-1 )), blas::Error );
//    assert_throw(( tmblas::omatcopy<TA,TB>( 'R',   m, n, alphat, At, lda, Bt, m-1 )), blas::Error );
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n",
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Bm ), llong( Bn ), llong( ldb ), llong( size_B ), Bnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha));
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Bm, Bn, B, ldb );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::omatcopy<TA,TB>( trans, m, n, alphat, At, lda, Bt, ldb );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_B; i++)
		B[i] = tmblas::type_conv<RB, TB>(Bt[i]);

    double gflop = Gflop_< scalar_t >::omatcopy( m, n );
    double gbyte = Gbyte_< scalar_t >::omatcopy( m, n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "B2 = " ); print_matrix( Bm, Bn, B, ldb );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
    	cblas_omatcopy( layout2char(layout), op2char(trans), m, n, alpha, A, lda, Bref, ldb );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "Bref = " ); print_matrix( Bm, Bn, Bref, ldb );
        }

        // error = ||Bref - B||
        cblas_axpy( size_B, -1.0, B, 1, Bref, 1 );
        real_t error = cblas_nrm2( size_B, Bref, 1 );
        params.error() = error;

        // omatcopy must be exact!
        params.okay() = (error == 0);
    }

    delete[] A;
    delete[] B;
    delete[] Bref;
    delete[] At;
    delete[] Bt;
}

// -----------------------------------------------------------------------------
void test_omatcopy( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_omatcopy_work< float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_omatcopy_work< double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_omatcopy_work< double, double, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_omatcopy_work< std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_omatcopy_work< std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_omatcopy_work< std::complex<double>, std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
