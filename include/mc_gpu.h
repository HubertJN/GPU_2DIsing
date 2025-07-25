#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include "gpu_tools.h"






// pre-compute acceptance probabilities for spin flips
void preComputeProbs_gpu(double beta, double h);

// pre-compute neighbours
void preComputeNeighbours_gpu(const int L, int *d_ising_grids, int *d_neighbour_list);


// MC sweep on the GPU - 3 versions
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);
__global__ void mc_sweep_gpu_bitrep(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);
__global__ void mc_sweep_gpu_bitmap32(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);
__global__ void mc_sweep_gpu_bitmap64(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);

// Compute magnetisation on the GPU
__global__ void compute_magnetisation_gpu(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation);

// Neighbour list squeezed into constant memory. Use of uint16_t limts MAXL to be 
// 128, or would need to move to 2D indexing.
//__constant__ uint16_t dc_neighbour_list[MAXL*MAXL*4];

#define MAXL 64
__constant__ uint8_t dc_next[MAXL];
__constant__ uint8_t dc_prev[MAXL];