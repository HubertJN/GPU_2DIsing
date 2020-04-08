#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include "gpu_tools.h"

// Cache of acceptance probabilities 
__constant__ float d_Pacc[20];   // gpu constant memory

// pre-compute acceptance probabilities for spin flips
void preComputeProbs_gpu(double beta, double h);

__global__ void mc_sweep_gpu(const int L, curandStatePhilox4_32_10_t *state, const int ngrids, int *d_ising_grids, const float beta, const float h);

