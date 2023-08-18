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
template <typename RA, typename RB, typename RC, typename TA, typename TB, typename TC>
void test_gemmt_work( Params& params, bool run )
{
	#if !defined (INTEL_MKL)
	printf ("Gemmt test requires Intel MKL.\n");
	exit (1);
	#endif
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Uplo;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RB, RC >;
    using scalar_tt = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    blas::Op transA = params.transA();
    blas::Op transB = params.transB();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    scalar_tt alphat = tmblas::type_conv<scalar_tt, scalar_t>(params.alpha());
    scalar_tt betat  = tmblas::type_conv<scalar_tt, scalar_t>(params.beta());
    int64_t n       = params.dim.n();
    int64_t k       = params.dim.k();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // setup
    int64_t Am = (transA == Op::NoTrans ? n : k);
    int64_t An = (transA == Op::NoTrans ? k : n);
    int64_t Bm = (transB == Op::NoTrans ? k : n);
    int64_t Bn = (transB == Op::NoTrans ? n : k);
    int64_t Cm = n;
    int64_t Cn = n;
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
        std::swap( Cm, Cn );
    }
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    int64_t ldc = roundup( Cm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    size_t size_C = size_t(ldc)*Cn;
    RA* A    = new RA[ size_A ];
    RB* B    = new RB[ size_B ];
    RC* C    = new RC[ size_C ];
    RC* Cref = new RC[ size_C ];
    TA* At   = new TA[ size_A ];
    TB* Bt   = new TB[ size_B ];
    TC* Ct   = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", Cm, Cn, C, ldc, Cref, ldc );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_B; i++)
		Bt[i] = tmblas::type_conv<TB, RB>(B[i]);
	for (size_t i = 0; i < size_C; i++)
		Ct[i] = tmblas::type_conv<TC, RC>(C[i]);

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::gemmt<TA,TB,TC>( Uplo(0), transA,  transB,  n,  k, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  Op(0),  transB,  n,  k, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  transA, Op(0),   n,  k, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  transA, transB, -1,  k, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  transA, transB,  n, -1, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  Op::NoTrans,   Op::NoTrans, n, k, alphat, At, n-1, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  Op::Trans,     Op::NoTrans, n, k, alphat, At, k-1, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  Op::ConjTrans, Op::NoTrans, n, k, alphat, At, k-1, Bt, ldb, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  Op::NoTrans, Op::NoTrans,   n, k, alphat, At, lda, Bt, k-1, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  Op::NoTrans, Op::Trans,     n, k, alphat, At, lda, Bt, n-1, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  Op::NoTrans, Op::ConjTrans, n, k, alphat, At, lda, Bt, n-1, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::gemmt<TA,TB,TC>( uplo,  transA, transB, n, k, alphat, At, lda, Bt, ldb, betat, Ct, n-1 )), blas::Error );
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C Cm=%5lld, Cn=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Bm ), llong( Bn ), llong( ldb ), llong( size_B ), Bnorm,
                llong( Cm ), llong( Cn ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Bm, Bn, B, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::gemmt<TA,TB,TC>( uplo,  transA, transB, n, k,
                alphat, At, lda, Bt, ldb, betat, Ct, ldc );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_C; i++)
		C[i] = tmblas::type_conv<RC, TC>(Ct[i]);

    double gflop = blas::Gflop< scalar_t >::gemm( n, n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_gemmt(cblas_layout_const(layout),
					cblas_uplo_const(uplo),
                    cblas_trans_const(transA),
                    cblas_trans_const(transB),
                    n, k, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, k, alpha, beta, Anorm, Bnorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
    delete[] At;
    delete[] Bt;
    delete[] Ct;
}

// -----------------------------------------------------------------------------
void test_gemmt( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gemmt_work< float, float, float, float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gemmt_work< double, double, double, double, double, double >( params, run );
            break;

        case testsweeper::DataType::Quadruple:
            test_gemmt_work< double, double, double, quadruple, quadruple, quadruple >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemmt_work< std::complex<float>, std::complex<float>, std::complex<float>,
            				 std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemmt_work< std::complex<double>, std::complex<double>, std::complex<double>,
            				 std::complex<double>, std::complex<double>, std::complex<double> >( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_gemmt_work< std::complex<double>, std::complex<double>, std::complex<double>,
            				 std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
