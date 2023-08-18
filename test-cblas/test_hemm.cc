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
void test_hemm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo;
    using blas::Side;
    using blas::Layout;
    using scalar_t = blas::scalar_type< RA, RB, RC >;
    using scalar_tt = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    scalar_tt alphat = tmblas::type_conv<scalar_tt, scalar_t>(params.alpha());
    scalar_tt betat  = tmblas::type_conv<scalar_tt, scalar_t>(params.beta());
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // setup
    int64_t An = (side == Side::Left ? m : n);
    int64_t Cm = m;
    int64_t Cn = n;
    if (layout == Layout::RowMajor)
        std::swap( Cm, Cn );
    int64_t lda = roundup( An, align );
    int64_t ldb = roundup( Cm, align );
    int64_t ldc = roundup( Cm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Cn;
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
    real_t Anorm = lapack_lansy( "f", uplo2str(uplo), An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Cm, Cn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

	for (size_t i = 0; i < size_A; i++)
		At[i] = tmblas::type_conv<TA, RA>(A[i]);
	for (size_t i = 0; i < size_B; i++)
		Bt[i] = tmblas::type_conv<TB, RB>(B[i]);
	for (size_t i = 0; i < size_C; i++)
		Ct[i] = tmblas::type_conv<TC, RC>(C[i]);

    // test error exits
	#if !defined (INTEL_MKL)
    assert_throw(( tmblas::hemm<TA,TB,TC>( Side(0),  uplo,     m,  n, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::hemm<TA,TB,TC>( side,     Uplo(0),  m,  n, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::hemm<TA,TB,TC>( side,     uplo,    -1,  n, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::hemm<TA,TB,TC>( side,     uplo,     m, -1, alphat, At, lda, Bt, ldb, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::hemm<TA,TB,TC>( Side::Left,  uplo,     m,  n, alphat, At, m-1, Bt, ldb, betat, Ct, ldc )), blas::Error );
    assert_throw(( tmblas::hemm<TA,TB,TC>( Side::Right, uplo,     m,  n, alphat, At, n-1, Bt, ldb, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::hemm<TA,TB,TC>( side, uplo,  m,  n, alphat, At, lda, Bt, m-1, betat, Ct, ldc )), blas::Error );

    assert_throw(( tmblas::hemm<TA,TB,TC>( side, uplo,  m,  n, alphat, At, lda, Bt, ldb, betat, Ct, m-1 )), blas::Error );
	#endif

    if (verbose >= 1) {
        printf( "\n"
                "side %c, uplo %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B  m=%5lld,  n=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C  m=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                side2char(side), uplo2char(uplo),
                llong( An ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( m ), llong( n ), llong( ldb ), llong( size_B ), Bnorm,
                llong( m ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( An, An, A, lda );
        printf( "B = "    ); print_matrix( Cm, Cn, B, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::hemm<TA,TB,TC>( side, uplo, m, n,
                alphat, At, lda, Bt, ldb, betat, Ct, ldc );
    time = get_wtime() - time;

	for (size_t i = 0; i < size_C; i++)
		C[i] = tmblas::type_conv<RC, TC>(Ct[i]);

    double gflop = blas::Gflop< scalar_t >::hemm( side, m, n );
    params.time()   = time;
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_hemm( cblas_layout_const(layout),
                    cblas_side_const(side),
                    cblas_uplo_const(uplo),
                    m, n, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, An, alpha, beta, Anorm, Bnorm, Cnorm,
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
void test_hemm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
        case testsweeper::DataType::Double:
        case testsweeper::DataType::Quadruple:

        case testsweeper::DataType::SingleComplex:
            test_hemm_work< std::complex<float>, std::complex<float>, std::complex<float>,
            				std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hemm_work< std::complex<double>, std::complex<double>, std::complex<double>,
            				std::complex<double>, std::complex<double>, std::complex<double> >( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_hemm_work< std::complex<double>, std::complex<double>, std::complex<double>,
            				std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
