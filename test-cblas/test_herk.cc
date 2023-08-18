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
template <typename RA, typename RC, typename TA, typename TC>
void test_herk_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Uplo;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RC >;
    using scalar_tt = blas::scalar_type< TA, TC >;
    using real_t   = blas::real_type< scalar_t >;
    using real_tt  = blas::real_type< scalar_tt >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    blas::Uplo uplo = params.uplo();
    real_t alpha    = params.alpha();  // note: real
    real_tt alphat  = tmblas::type_conv<real_tt, real_t>(params.alpha());
    real_t beta     = params.beta();   // note: real
    real_tt betat   = tmblas::type_conv<real_tt, real_t>(params.beta());
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
    int64_t Am = (trans == Op::NoTrans ? n : k);
    int64_t An = (trans == Op::NoTrans ? k : n);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda = roundup( Am, align );
    int64_t ldc = roundup(  n, align );
    size_t size_A = size_t(lda)*An;
    size_t size_C = size_t(ldc)*n;
    RA* A    = new RA[ size_A ];
    RC* C    = new RC[ size_C ];
    RC* Cref = new RC[ size_C ];
    TA* At   = new TA[ size_A ];
    TC* Ct   = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", n, n, C, ldc, Cref, ldc );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Cnorm = lapack_lansy( "f", uplo2str(uplo), n, C, ldc, work );

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_C; i++)
		Ct[i] = tmblas::type_conv<TC, RC>(C[i]);

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::herk<TA,TC>( Uplo(0), trans,  n,  k, alphat, At, lda, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::herk<TA,TC>( uplo,    Op(0),  n,  k, alphat, At, lda, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::herk<TA,TC>( uplo,    trans, -1,  k, alphat, At, lda, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::herk<TA,TC>( uplo,    trans,  n, -1, alphat, At, lda, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::herk<TA,TC>( uplo, Op::NoTrans,   n, k, alphat, At, n-1, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::herk<TA,TC>( uplo, Op::Trans,     n, k, alphat, At, k-1, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::herk<TA,TC>( uplo, Op::ConjTrans, n, k, alphat, At, k-1, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::herk<TA,TC>( uplo,    trans,  n,  k, alphat, At, lda, betat, Ct, n-1 )), blas::Error );

    if (blas::is_complex<scalar_t>::value) {
        // complex herk doesn't allow Trans, only ConjTrans
        assert_throw(( tmblas::herk<TA,TC>( uplo, Op::Trans, n, k, alphat, At, lda, betat, Ct, ldc )), blas::Error );
    }
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "uplo %c, trans %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "C  n=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                uplo2char(uplo), op2char(trans),
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( n ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e; beta = %.4e;  %% real\n", alpha, beta );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "C = "    ); print_matrix(  n,  n, C, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::herk<TA,TC>( uplo, trans, n, k,
                alphat, At, lda, betat, Ct, ldc );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_C; i++)
		C[i] = tmblas::type_conv<RC, TC>(Ct[i]);

    double gflop = blas::Gflop< scalar_t >::herk( n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( n, n, C, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_herk( cblas_layout_const(layout),
                    cblas_uplo_const(uplo),
                    cblas_trans_const(trans),
                    n, k, alpha, A, lda, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( n, n, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_herk( uplo, n, k, alpha, beta, Anorm, Anorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] C;
    delete[] Cref;
    delete[] At;
    delete[] Ct;
}

// -----------------------------------------------------------------------------
void test_herk( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
        case testsweeper::DataType::Double:
        case testsweeper::DataType::Quadruple:

        case testsweeper::DataType::SingleComplex:
            test_herk_work< std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_herk_work< std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_herk_work< std::complex<double>, std::complex<double>, std::complex<quadruple>, std::complex<quadruple> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
