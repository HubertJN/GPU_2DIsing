#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Where you want the binary
prefix     = $(shell pwd)
bindir     = $(prefix)/bin
srcdir     = $(prefix)/src
incdir     = $(prefix)/include
objdir     = $(prefix)/obj

# Create bin/ and obj/ if they do not exist
$(shell mkdir -p $(bindir))
$(shell mkdir -p $(objdir))
$(shell mkdir -p out)

CC    = gcc
NVCC  = nvcc
LD    = nvcc
CFLAGS = -O3 -I$(incdir)
NVFLAGS = -O3 -gencode arch=compute_86,code=sm_86 --generate-line-info -I$(incdir)

# Define objects in dependency order
OBJECTS = mt19937ar.o gpu_tools.o mc_cpu.o mc_gpu.o io.o
OBJECTS := $(addprefix $(objdir)/, $(OBJECTS))

.PRECIOUS: %.o
.PHONY: clean all

all: GPU_2DIsing

$(objdir)/%.o: $(srcdir)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(objdir)/%.o: $(srcdir)/%.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<

GPU_2DIsing: $(OBJECTS) $(srcdir)/ising.cu
	$(LD) -o $(bindir)/GPU_2DIsing $(OBJECTS) $(srcdir)/ising.cu $(NVFLAGS) -lhdf5

clean:
	rm -f $(objdir)/*.o
	rm -f $(bindir)/GPU_2DIsing

