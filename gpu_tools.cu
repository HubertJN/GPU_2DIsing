// -*- mode: C -*-

#include <stdio.h>
#include "gpu_tools.h"

cudaError_t err;  // cudaError_t is a type defined in cuda.h


// Boilerplate error checking code borrowed from stackoverflow
void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

// Initialisation 
void gpuInit(int deviceIndex){

    int idev, count;    

    // Make sure we have a CUDA capable device to work with
    err = cudaGetDeviceCount(&count);
    if ( (count==0) || (err!=cudaSuccess) ) {
        fprintf(stderr,"No CUDA supported devices are available in this system.\n");
        exit(EXIT_FAILURE);
    } else {
        //fprintf(stderr,"Found %d CUDA devices in this system\n",count);
    }


    // cudaDeviceProp is a type of struct which we will
    // populate with information about the available 
    // GPU compute devices.
    cudaDeviceProp prop;

    // Loop over the available CUDA devices
    for (idev=0;idev<count;idev++) {

        // Call another CUDA helper function to populate prop
        gpuErrchk( cudaGetDeviceProperties(&prop,idev) );

        // Print out a member of the prop struct which tells
        // us the name of the CUDA device. Other members of this
        // struct tell us the clock speed and compute capability
        // of the device.
        /*
        fprintf(stderr,"Device %d : %s\n",idev,prop.name);
        fprintf(stderr,"================================\n");
        fprintf(stderr,"Number of SMs       : %d\n",prop.multiProcessorCount);
        fprintf(stderr,"Max SHMEM per block : %ld KB\n",prop.sharedMemPerBlock/1024);
        //fprintf(stderr,"Warp size           : %d\n",prop.warpSize);
        //fprintf(stderr,"Global DRAM         : %ld\n",prop.totalGlobalMem);
        fprintf(stderr,"\n");
        */
    }


    // Only set device if specified 
    if (deviceIndex != -1) {
        gpuErrchk( cudaSetDevice(deviceIndex) );
    }

    gpuErrchk( cudaGetDevice(&idev ) );
    //fprintf(stderr,"Using CUDA device : %d\n",idev);

}


__global__ void populate_random(int length, float *rnd_array, curandStatePhilox4_32_10_t *state){


    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length){

        // 4 random numbers
        float4 rnd = curand_uniform4(&state[idx]);

        // use one of these
        rnd_array[idx] = rnd.z;

    }       

    return;

}



// Kernel to initialise RNG on the GPU. Used the cuRAND device API with one
// RNG sequence per CUDA thread.
__global__ void init_gpurand(unsigned long long seed, int ngrids, curandState *state){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx<ngrids){

        unsigned long long seq = (unsigned long long)idx;
        
        // Seperate subsequence for each thread
        curand_init(seed, seq, 0ull, &state[idx]);
    
        // Different seed for each thread (faster but risky)
        //curand_init(seed+23498*idx, 0ull, 0ull, &state[idx]);
    }


  }