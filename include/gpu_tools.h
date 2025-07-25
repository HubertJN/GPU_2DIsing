#ifndef GPU_TOOLS_H
#define GPU_TOOLS_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Prototype for function which initialises gpu
void gpuInit(int deviceIndex=-1);

// Macro for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Prototype for gpuAssert
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

// Prototype for RNG initialisation
__global__ void init_gpurand(unsigned long long seed, int ngrids, curandState *state);

// For testing generation of random numbers on each thread
__global__ void populate_random(int length, float *rnd_array, curandState *state);

#endif