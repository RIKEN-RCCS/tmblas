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
template <typename Tbase, typename RA, typename RB, typename RC, typename TA, typename TB, typename TC>
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

    bool trunc = true;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
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
    int64_t align   = params.align();
    //    int64_t verbose = params.verbose();

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

    Tbase* A    = new Tbase[ size_A ];
    Tbase* B    = new Tbase[ size_B ];
    Tbase* C    = new Tbase[ size_C ];
    
    RA* Ar    = new RA[ size_A ];
    RB* Br    = new RB[ size_B ];
    RC* Cr    = new RC[ size_C ];

    TA* At   = new TA[ size_A ];
    TB* Bt   = new TB[ size_B ];
    TC* Ct   = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_A, A, trunc);
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_B, B, trunc);
    tmblas_larnv<blas::real_type<Tbase> >( idist, iseed, size_C, C, trunc);
    //    lapack_lacpy( "g", Cm, Cn, C, ldc, Cref, ldc );

    // norms for error check
#if 0
    real_t work[1];
    real_t Anorm = lapack_lansy( "f", uplo2str(uplo), An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Cm, Cn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );
#endif
    for (size_t i = 0; i < size_A; i++) {
      At[i] = tmblas::type_conv<TA, Tbase>(A[i]);
      Ar[i] = tmblas::type_conv<RA, Tbase>(A[i]);      
    }
    for (size_t i = 0; i < size_B; i++) {
      Bt[i] = tmblas::type_conv<TB, Tbase>(B[i]);
      Br[i] = tmblas::type_conv<RB, Tbase>(B[i]);      
    }
    for (size_t i = 0; i < size_C; i++) {
      Ct[i] = tmblas::type_conv<TC, Tbase>(C[i]);
      Cr[i] = tmblas::type_conv<RC, Tbase>(C[i]);      
    }
#if 0
    // test error exits
    assert_throw( tmblas::hemm( Side(0),  uplo,     m,  n, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::hemm( side,     Uplo(0),  m,  n, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::hemm( side,     uplo,    -1,  n, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::hemm( side,     uplo,     m, -1, alphat, At, lda, Bt, ldb, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::hemm( Side::Left,  uplo,     m,  n, alphat, At, m-1, Bt, ldb, betat, Ct, ldc ), blas::Error );
    assert_throw( tmblas::hemm( Side::Right, uplo,     m,  n, alphat, At, n-1, Bt, ldb, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::hemm( side, uplo,  m,  n, alphat, At, lda, Bt, m-1, betat, Ct, ldc ), blas::Error );

    assert_throw( tmblas::hemm( side, uplo,  m,  n, alphat, At, lda, Bt, ldb, betat, Ct, m-1 ), blas::Error );

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
#endif
    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    tmblas::hemm<TA, TB, TC>( side, uplo, m, n,
                alphat, At, lda, Bt, ldb, betat, Ct, ldc );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::hemm( side, m, n );
    params.time()   = time;
    params.gflops() = gflop / time;
#if 0
    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C, ldc );
    }
#endif
    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
	blas::hemm<RA, RB, RC>( layout, side, uplo,
                    m, n, alphar, Ar, lda, Br, ldb, betar, Cr, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;
#if 0
        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }
#endif
        // check error compared to reference
        double errormax, errorl2;	
        bool okay = true;
	//       check_gemm( Cm, Cn, An, alpha, beta, Anorm, Bnorm, Cnorm,
	//                    Cref, ldc, C, ldc, verbose, &error, &okay );
	check_diff<RC, TC>(Cm, Cn, Cr, Ct, ldc, &errormax, &errorl2);
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
    delete[] B;
    delete[] C;
    delete[] Ar;
    delete[] Br;
    delete[] Cr;    
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
    case testsweeper::DataType::Octuple:
      
        case testsweeper::DataType::SingleComplex:
	  test_hemm_work< std::complex<float>,
			  std::complex<float>, std::complex<float>, std::complex<float>,
			  std::complex<float>, std::complex<float>, std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hemm_work< std::complex<double>,
			    std::complex<double>, std::complex<double>, std::complex<double>,
			    std::complex<double>, std::complex<double>, std::complex<double> >( params, run );
            break;

        case testsweeper::DataType::QuadrupleComplex:
            test_hemm_work< std::complex<double>,
			    std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple>,
			    std::complex<quadruple>, std::complex<quadruple>, std::complex<quadruple> >( params, run );
            break;

        case testsweeper::DataType::OctupleComplex:
            test_hemm_work< std::complex<double>,
			    std::complex<octuple>, std::complex<octuple>, std::complex<octuple>,
			    std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
