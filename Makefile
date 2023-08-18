# Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

all:
	cd src/; make -f Makefile.generic -j; cd ..;
	cd test-self; make -f Makefile.generic -j; cd ..;
	cd test-cblas; make -f Makefile.generic -j; cd ..;
	cd test-blaspp; make -f Makefile.generic -j; cd ..;
clean:
	cd src/; make -f Makefile.generic clean; cd ..;
	cd test-self; make -f Makefile.generic clean; cd ..;
	cd test-cblas; make -f Makefile.generic clean; cd ..;
	cd test-blaspp; make -f Makefile.generic clean; cd ..;
