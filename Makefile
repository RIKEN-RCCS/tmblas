# Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

.PHONY: all clean

all:
	$(MAKE) -C src -f Makefile.generic
	$(MAKE) -C test-self -f Makefile.generic
	$(MAKE) -C test-cblas -f Makefile.generic
	$(MAKE) -C test-blaspp -f Makefile.generic
clean:
	$(MAKE) -C src -f Makefile.generic $@
	$(MAKE) -C test-self -f Makefile.generic $@
	$(MAKE) -C test-cblas -f Makefile.generic $@
	$(MAKE) -C test-blaspp -f Makefile.generic $@
