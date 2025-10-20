#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Where you want the binary (set to current working directory)
prefix     = $(CURDIR)
bindir     = $(prefix)/bin

# Object directory
objdir = obj

# Define objects with .obj extension in dependency order
OBJECTS = $(objdir)/mt19937ar.obj $(objdir)/grid.obj $(objdir)/gpu_tools.obj $(objdir)/io.obj $(objdir)/mc_cpu.obj $(objdir)/mc_gpu.obj $(objdir)/bootstrap.obj

CC    = gcc
NVCC  = nvcc
LD    = nvcc
CFLAGS =  -O3 -g

# HDF5 flags (use pkg-config if available)
HDF5_CFLAGS := $(shell pkg-config --cflags hdf5 2>/dev/null)
HDF5_LIBS   := $(shell pkg-config --libs hdf5 2>/dev/null)

# Append HDF5 cflags to C compile flags so headers are found
CFLAGS += $(HDF5_CFLAGS)

NVFLAGS = -O3 -gencode arch=compute_61,code=sm_61 --generate-line-info  -Wno-deprecated-gpu-targets # Quadro P2000 in Telamon

.PRECIOUS: $(objdir)/%.obj
.PHONY:  clean all

all : gasp

# Create object directory if it does not exist
$(objdir):
	mkdir -p $(objdir)

# Create bin directory if it does not exist
$(bindir):
	mkdir -p $(bindir)

# Compile C files to obj
$(objdir)/%.obj: src/%.c include/%.h | $(objdir)
	$(CC) $(CFLAGS) -c -o $@ $< -Iinclude/ $(HDF5_CFLAGS)

# Compile CUDA files to obj
$(objdir)/%.obj: src/%.cu include/%.h | $(objdir)
	$(NVCC) $(NVFLAGS) -c -o $@ $< -Iinclude/ $(HDF5_CFLAGS)

gasp : $(OBJECTS) src/ising.cu | $(objdir) $(bindir)
	$(LD) -o $(bindir)/gasp $(OBJECTS) src/ising.cu $(NVFLAGS) -Iinclude/ $(HDF5_LIBS)

clean :
	rm -rf $(objdir)
	rm -f $(bindir)/GPU_2DIsing
	rm -f *.mod *.d *.il work.*

