# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 4

test_sequential:
	tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(MPICC) tester.c knnring_mpi.a -o $@ -lm -lopenblas -lm
	./test_sequential


test_mpi:
	tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(MPICC) -g tester_mpi.c knnring_mpi.a -o $@ -lopenblas -lm 
	$(MPIRUN) ./test_mpi

