#!/bin/bash

make clean -f Makefile_GPU
make -f Makefile_GPU
make clean -f Makefile_comm
make -f Makefile_comm
make clean -f Makefile_cluster
make -f Makefile_cluster
