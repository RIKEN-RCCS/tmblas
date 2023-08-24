# tmBLAS (version 0.1)

<h2>Overview</h2>
<p><b>Templated mixed-precision Basic Linear Algebra Subprograms (tmBLAS)</b> is a reference BLAS implementation for mixed-precision computation implemented using C++ template based on BLAS++. For mixed-precision computation, tmBLAS decouples data-types of operator and operands in BLAS routines; each operand can take a different data-type, and it can perform operations with higher precision than the operands' data-type.</p>
<p>tmBLAS is a template library, which you can instantiate with any data type you want, but has already instantiated routines with half, float, double, quadruple, and octuple-precision input/output data and operations, which also support operations in one level higher precision than the data precision.</p>
<p>For decoupling data-types of operator and operands, some routines require an additional working array to store intermediate values. Users can select to prepare the working array prior to calling routine or to ask the routine for dynamic allocation and deallocation of them.</p>

<h2>Instantiated data-types</h2>
<p>Mono-precision and mixed-precision operations in one level higher precision than the data precision for the following data-types are instantiated.</p>
<ul>
<li>half-precision as half (binary16)</li>
<li>single-precision as float (binary32)</li>
<li>double-precision as double (binary64)</li>
<li>quadruple-precision as dd_real (using QD library, 106-bit mantissa) or mpfr128 (using MPFR, 113-bit mantissa, equivalent to binary128)</li>
<li>octuple-precision as qd_real (using QD library, 212-bit mantissa) or mpfr256 (using MPFR, 237-bit mantissa, equivalent to binary256)</li>
</ul>

<h2>Routines</h2>
<h3>LEVEL 1</h3>
<ul>
<li>scal: x = a * x</li>
<li>axpy: y = a * x + y</li>
<li>dot: dot product // conjg is only used for complex data</li>
<li>nrm2: Euclidean norm // non-unit stride mixedpmul -> madd</li>
<li>iamax: index of max abs value // without mixed precision arithmetic</li>
</ul>
<h3>LEVEL 2</h3>
<ul>
<li>gemv: matrix vector multiply</li>
<li>symv: symmetric matrix vector multiply</li>
<li>hemv: Hermitian version of symv</li>
<li>trmv: triangular matrix vector multiply</li>
<li>trsv: solving triangular matrix problems</li>
<li>ger: rank 1 operation A := alpha * x * y' + A</li>
<li>geru: rank 1 operation A := alpha * x * y^T + A</li>
<li>syr: symmetric rank 1 operation A = alpha * x * x' + A</li>
<li>her: Hermitian version of syr with "real" alpha</li>
<li>syr2: symmetric rank 2 operation A = alpha * x * y' + alpha * y * x' + A</li>
<li>her2: Hermitian version of syr2</li>
</ul> 
<h3>LEVEL 3</h3>
<ul>
<li>gemm: matrix matrix multiply</li>
<li>symm: symmetric matrix matrix multiply</li>
<li>hemm: Hermitian version of symm</li>
<li>syrk: symmetric rank-k update to a matrix</li>
<li>herk: Hermitian version of syrk</li>
<li>syr2k: symmetric rank-2k update to a matrix</li>
<li>her2k: Hermitian version of syr2k</li>
<li>trmm: triangular matrix matrix multiply</li>
<li>trsm: solving triangular matrix with multiple right hand sides</li>
</ul> 
<h3>Extension</h3>
<ul>
<li>gemmt: synmmetric result of matrix matrix multiply</li>
<li>omatcopy: block copy of the matrix to the out place</li>
<li>csrgemv: sparse matrix vector multiply (CSR format)</li>
<li>csrsymv: sparse symmetric matrix vector multiply (CSR format)</li>
<li>csrhemv: sparse Hermitian matrix vetor multiply (CSR format)</li>
<li>csrgemm: sparse matrix matrix multiply (CSR format)</li>
<li>csrsymm: sparse symmetric matrix matrix multiply (CSR format)</li>
<li>csrhemm: sparse Hermitian matrix matrix multiply (CSR format)</li>
</ul>
<p>Note: the mixture of real and complex is not supported.</p>
<h2>Requirements</h2>
<ul>
<li>C++ compiler with C++11 (gcc, clang, icc)</li>
<li>BLAS++ (for test): https://github.com/icl-utk-edu/blaspp</li>
<li>BLAS and CBLAS (for test)</li>
<li>LAPACK and MKL (for test)</li>
<li>QD (for quadruple and octuple): https://www.davidhbailey.com/dhbsoftware/</li>
<li>MPFR (for quadruple and octuple)
<li>half.hpp (half precision emulator): https://github.com/melowntech/half/</li>
<li>real.hpp (C++ wrapper for MPFR): http://chschneider.eu/programming/mpfr_real/</li>
</ul>

<h2>Downloads (tar ball)</h2>
<ul>
<li><a href="https://www.r-ccs.riken.jp/labs/lpnctrt/projects/tmblas/">https://www.r-ccs.riken.jp/labs/lpnctrt/projects/tmblas/</a></li>
</ul>

<h2>How to build</h2>
<ol>
  <li>Modify "Makefile.generic.inc"</li>
  <li>Place "half.hpp" and "real.hpp" in src/</li>
  <li>$ make</li>
</ol>

<h2>Test</h2>
<p>The algorithms written as templates were validated in the following way: we confirmed that the results were bit-level consistent with BLAS++, using floating-point inputs that were truncated to avoid rounding errors in double precision operations. These tests were performed because some of the routines changed the loop order from the BLAS++ code, which changed the order of the floating-point operations, resulting in a loss of reproducibility. For TRSV and TRSM, we use reduced data obtained by factorization in LAPACK as the same way of BLAS++ and truncate them as other routines. Testing for the accuracy of the mixed-precision routines are still under consideration.</p>
<ul>
<li>test-cblas: blaspp's test, comparing with CBLAS (MKL) in single and double precision</li>
<li>test-blaspp: check bit-level consistency of results with blaspp's templated codes (included) in double, quadruple, and octuple precision using truncated floating-point input</li>
<li>test-self: tests for arithmetic operators and sparse routines</li>
</ul>

<h2>Publications</h2>
<ul>
<li>Atsushi Suzuki, Daichi Mukunoki, Toshiyuki Imamura, tmBLAS: a Mixed Precision BLAS by C++ Template, ISC High Performance (ISC 2023), research poster session, May, 2023.</li>
<li>Daichi Mukunoki, Atsushi Suzuki, Toshiyuki Imamura, Multiple and Mixed Precision BLAS with C++ Template, 5th R-CCS International Symposium, Feb. 6, 2023.</li>
</ul>

<h2>Contact</h2>
<ul>
<li>Atsushi Suzuki, R-CCS</li>
<li>Daichi Mukunoki, R-CCS</li>
<li>Toshiyuki Imamura, R-CCS</li>
<li>Large-scale Parallel Numerical Computing Technology Research Team (https://www.r-ccs.riken.jp/labs/lpnctrt/index.html), RIKEN Center for Computational Science</li>
</ul>


