# Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.	
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

UNAME := $(shell uname -s)

# BLAS for tmBLAS
BLAS = MKL

# BLAS for test-cblas 
TESTBLAS = MKL
# Omatcopy and gemmt require MKL

QUAD = QD_LIBRARY
#QUAD = MPFR
HALF = HALF_EMULATION
WORK_DIR = $(HOME)

MKL_INCDIR = /opt/intel/oneapi/mkl/2023.0.0/include
MKL_LIBDIR = /opt/intel/oneapi/mkl/2023.0.0/lib/intel64

CXXFLAGS = -std=c++11 -O2 -Wall

ifeq ($(UNAME), Linux)
  CXX = icpx -fp-model=precise
# CXX = icpc -diag-disable=10441 -fp-model=precise
# CXX = g++
  AR  = ar cr 
ifeq ($(BLAS), MKL)
  CXXFLAGS += -DINTEL_MKL #-DMKL_OMATCOPY
  BLAS_INCDIR = $(MKL_INCDIR)
  BLAS_LIBDIR = $(MKL_LIBDIR)
  BLASINC = -I$(BLAS_INCDIR)
  BLASLIB = -Xlinker -rpath=$(BLAS_LIBDIR) -L$(BLAS_LIBDIR) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
else ifeq ($(strip $(BLAS)),)
  BLASINC =
  BLASLIB = #-lcblas -lblas -lgfortran 
endif

ifeq ($(QUAD), QD_LIBRARY)
  CXXFLAGS += -DQD_LIBRARY
  QD_INCDIR = $(WORK_DIR)/include
  QD_LIBDIR= $(WORK_DIR)/lib
  QUADINC = -I$(QD_INCDIR)
  QUADLIB = -Xlinker -rpath=$(QD_LIBDIR) -L$(QD_LIBDIR) -lqd
else ifeq ($(QUAD), MPFR)
  CXXFLAGS += -DMPFR
  MPFR_INCDIR = $(WORK_DIR)/include
  MPFR_LIBDIR = $(WORK_DIR)/lib
  QUADINC = -I$(MPFR_INCDIR)
  QUADLIB = -Xlinker -rpath=$(MPFR_LIBDIR) -L$(MPFR_LIBDIR) -lmpfr
else
  QUADINC =
  QUADLIB =  -lquadmath # not for clang
endif

endif 

ifeq ($(HALF), HALF_EMULATION)
  CXXFLAGS += -DHALF_EMULATION
else ifeq ($(HALF), HALF_HARDWARE)
  CXXFLAGS += -DHALF_HARDWARE
endif

## for test-blaspp and test-cblas
## here path is defined relatively to test-{blaspp,cblas}/ 

ifeq ($(TESTBLAS), MKL)
  CXXFLAGS += -DINTEL_MKL_TEST
  TESTBLASINC = -I$(MKL_INCDIR)
  TESTBLASLIB = -Xlinker -rpath=$(MKL_LIBDIR) -L$(MKL_LIBDIR) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
else ifeq ($(strip $(TESTBLAS)),)
  TESTBLAS_INC =
  TESTBLAS_LIB = #-llapacke -llapack -lcblas -lblas -lgfortran
endif

ifeq ($(TESTBLASLIB), $(BLASLIB))
  TESTBLASLIB =
endif

TMBLAS_INCDIR = ../src
TMBLAS_LIBDIR = ../src 
BLASPP_INCDIR = ../blaspp/include/
TESTSWEEPER = ../testsweeper

TESTSWEEPERLIB = -Wl,-rpath,$(TESTSWEEPER) -L$(TESTSWEEPER) -ltestsweeper
TMBLASINC = -I$(TMBLAS_INCDIR)
TMBLASLIB = -L$(TMBLAS_LIBDIR) -ltmblas
BLASPPINC = -I$(BLASPP_INCDIR)
