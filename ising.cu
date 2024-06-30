// -*- mode: C -*-
/* ==========================================================================================
                                 GPU_2DIsing.cu

Implementation of the 2D Ising model in CUDA. Each CUDA thread simulates an independent 
instance of the 2D Ising model in parallel with an independent random number sequence. Draws
heavily from the work of Weigel et al, [J. Phys.: Conf. Ser.921 012017 (2017)] but used here
for gathering rare event statistics on nucleation during magnetisation reversal. 
 ===========================================================================================*/
// D. Quigley. Univeristy of Warwick

// TODO
// 1. sweep counter probably needs to be a long and not an int
// 2. clustering on CPU asynchronously with GPU ?
// 3. write magnetisation to binary file ?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>

extern "C" {
  #include "mc_cpu.h"
  #include "io.h"
  #include "functions/read_input_variables.h"
}

#include "mc_gpu.h"
#include "gpu_tools.h"

const bool run_gpu = true;      // Run using GPU
const bool run_cpu = false;     // Run using CPU

int main (int argc, char *argv[]) {

/*=================================
   Constants and variables
  =================================*/ 
  
  int L       = 64;            // Size of 2D Ising grid. LxL grid squares.
  int nreplicas  = 1;             // Number of replicas of 2D grid to simulate
  int nsweeps = 100;       // Total number of MC sweeps to simulate on each grid

  int itask = 0;               // 0 = count samples which nucleate, 1 = compute committor

  int mag_output_int  = 100;   // Number of MC sweeps between calculation of magnetisation
  int grid_output_int = 1000;  // Number of MC sweeps between dumps of grid to file

  double beta = 0.54;       // Inverse temperature
  double h = 0.07;          // External field

  double dn_threshold = -0.90;         // Magnetisation at which we consider the system to have reached spin up state
  double up_threshold =  0.90;         // Magnetisation at which we consider the system to have reached spin down state

  unsigned long rngseed = 0;  // RNG seed (fixed for development/testing)
  //unsigned long rngseed = (long)time(NULL);

  int threadsPerBlock = 32;            // Number of threads/replicas to run in each threadBlock
  int blocksPerGrid   = 1;             // Total number of threadBlocks
  int gpu_device = -1;                 // GPU device to use
  int gpu_method = 0;                  // Which MC sweep kernel to use

/*=================================
   Read input file 
  =================================*/ 
  read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

/*=================================
   Delete old output 
  ================================*/
  remove("gridstates.bin");


/*=================================
   Write output header 
  ================================*/
  if (itask==0) {
    printf("# isweep    nucleated fraction\n");
  }

/*=================================
   Initialise simulations
  =================================*/ 
  // Host copy of Ising grid configurations
  int *ising_grids = (int *)malloc(L*L*nreplicas*sizeof(int));
  if (ising_grids==NULL){
    fprintf(stderr,"Error allocating memory for Ising grids!\n");
    exit(EXIT_FAILURE);
  }
  

  int i;
  int *grid_fate;  // stores pending(-1), reached B first (1) or reached A first (0)
  double pB;

  if (itask==0) {  // counting nucleated samples over time

    // Initialise as spin down  
    for (i=0;i<L*L*nreplicas;i++) { ising_grids[i] = -1; }

  } else if (itask==1) {

    // Read from file
    read_input_grid(L, nreplicas, ising_grids);

    grid_fate = (int *)malloc(nreplicas*sizeof(int));
    if (grid_fate==NULL) {
      printf("Error allocating memory for grid fates\n");
      exit(EXIT_FAILURE);
    }
    for (i=0;i<nreplicas;i++) { grid_fate[i] = -1; } // all pending

  } else {

    printf("Error - unknown value of itask!");
    exit(EXIT_FAILURE);

  }


  // TODO - replace with configuration read from file

  // Initialise host RNG
  init_genrand(rngseed);

  // Precompute acceptance probabilities for flip moves
  preComputeProbs_cpu(beta, h);

  int *d_ising_grids;                    // Pointer to device grid configurations
  curandState *d_state;                  // Pointer to device RNG states
  int *d_neighbour_list;                 // Pointer to device neighbour lists

  // How many sweeps to run in each call
  int sweeps_per_call;
  sweeps_per_call = mag_output_int < grid_output_int ? mag_output_int : grid_output_int;

  if (run_gpu==true) {
    
    gpuInit(gpu_device); // Initialise GPU device(s)

    // Allocate threads to thread blocks
    blocksPerGrid = nreplicas/threadsPerBlock;
    if (nreplicas%threadsPerBlock!=0) { blocksPerGrid += 1; }

    // Device copy of Ising grid configurations
    gpuErrchk( cudaMalloc(&d_ising_grids,L*L*nreplicas*sizeof(int)) );

    // Populate from host copy
    gpuErrchk( cudaMemcpy(d_ising_grids,ising_grids,L*L*nreplicas*sizeof(int),cudaMemcpyHostToDevice) );

    // Initialise GPU RNG
    gpuErrchk (cudaMalloc((void **)&d_state, nreplicas*sizeof(curandState)) );
    unsigned long long gpuseed = (unsigned long long)rngseed;
    init_gpurand<<<blocksPerGrid,threadsPerBlock>>>(gpuseed, nreplicas, d_state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    fprintf(stderr, "threadsPerBlock = %d, blocksPerGrid = %d\n",threadsPerBlock, blocksPerGrid);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);

    // Neighbours
    gpuErrchk (cudaMalloc((void **)&d_neighbour_list, L*L*4*sizeof(int)) );
    preComputeNeighbours_gpu(L, d_ising_grids, d_neighbour_list);

    // Test CUDA RNG (DEBUG)
    /*
    float   *testrnd = (float *)malloc(nreplicas*sizeof(float));
    float *d_testrnd;
    gpuErrchk( cudaMalloc(&d_testrnd, nreplicas*sizeof(float)) );

    int trial;
    for (trial=0;trial<10;trial++){

      populate_random<<<blocksPerGrid,threadsPerBlock>>>(nreplicas, d_testrnd, d_state);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      gpuErrchk( cudaMemcpy(testrnd, d_testrnd, nreplicas*sizeof(float), cudaMemcpyDeviceToHost) );

      for (i=0;i<nreplicas;i++){
        printf("Random number on grid %d : %12.4f\n",i,testrnd[i]);
      }
  
  }

    free(testrnd);
    cudaFree(d_testrnd);
    exit(EXIT_SUCCESS);
    */

  }

/*=================================
    Run simulations - CPU version
  =================================*/ 

  clock_t t1,t2;  // For measuring time taken
  int isweep;     // MC sweep loop counter
  int igrid;      // counter for loop over replicas



  if (run_cpu==true) {

    // Magnetisation of each grid
    double *magnetisation = (double *)malloc(nreplicas*sizeof(double));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation array!\n");
      exit(EXIT_FAILURE);
    }

    t1 = clock();  // Start timer

    isweep = 0;
    while (isweep < nsweeps){

      // Output grids to file
      if (isweep%grid_output_int==0){
        write_ising_grids(L, nreplicas, ising_grids, isweep);  
      }

      // Report magnetisations
      if (isweep%mag_output_int==0){
        for (igrid=0;igrid<nreplicas;igrid++){
          compute_magnetisation_cpu(L, ising_grids, igrid, magnetisation);
          //printf("Magnetisation of grid %d at sweep %d = %8.4f\n",igrid, isweep, magnetisation[igrid]);
        }
        if ( itask == 0 ) { // Report how many samples have nucleated.
          int nnuc = 0;
          for (igrid=0;igrid<nreplicas;igrid++){
            if ( magnetisation[igrid] > up_threshold ) nnuc++;
          }
          printf("%10d  %12.6f\n",isweep, (double)nnuc/(double)nreplicas);
          if (nnuc==nreplicas) break; // Stop if everyone has nucleated
        } else if ( itask == 1 ){

          // Statistics on fate of trajectories
          int nA=0, nB=0;
          for (igrid=0;igrid<nreplicas;igrid++){
            //printf("grid_fate[%d] = %d\n",igrid, grid_fate[igrid]);
            if (grid_fate[igrid]==0 ) {
              nA++;
            } else if (grid_fate[igrid]==1 ) {
              nB++;
            } else {
              if ( magnetisation[igrid] > up_threshold ){
                grid_fate[igrid] = 1;
                nB++;
              } else if (magnetisation[igrid] < dn_threshold ){
                grid_fate[igrid] = 0;
                nA++;
              }
            } // fate
          } //grids

          // Monitor progress
          pB = (double)nB/(double)(nA+nB);
          printf("\r Sweep : %10d, Reached m = %6.2f : %4d , Reached m = %6.2f : %4d , Unresolved : %4d, pB = %10.6f",
           isweep, dn_threshold, nA, up_threshold, nB, nreplicas-nA-nB,pB);
          fflush(stdout);
          if (nA + nB == nreplicas) break; // all fates resolved
        } // task
      } 

      // MC Sweep - CPU
      for (igrid=0;igrid<nreplicas;igrid++) {
        mc_sweep_cpu(L, ising_grids, igrid, beta, h, sweeps_per_call);
      }
      isweep += sweeps_per_call;

    }

    t2 = clock();  // Stop Timer

    printf("\n# Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);
    if (itask==1) { printf("pB estimate : %10.6f\n",pB); }

    // Release memory
    free(magnetisation);

  }

  /*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){

    // Host copy of magnetisation
    float *magnetisation = (float *)malloc(nreplicas*sizeof(float));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation host array!\n");
      exit(EXIT_FAILURE);
    }

    // Device copy of magnetisation
    float *d_magnetisation;
    gpuErrchk( cudaMalloc(&d_magnetisation,nreplicas*sizeof(float)) );

    // Streams
    cudaStream_t stream1;
    gpuErrchk( cudaStreamCreate(&stream1) );

    cudaStream_t stream2;
    gpuErrchk( cudaStreamCreate(&stream2) );


    t1 = clock();  // Start Timer

    isweep = 0;
    while(isweep < nsweeps){



      // Output grids to file
      if (isweep%grid_output_int==0){
        // Asynchronous - can happen while magnetisation is being computed in stream 2
        gpuErrchk( cudaMemcpyAsync(ising_grids,d_ising_grids,L*L*nreplicas*sizeof(int),cudaMemcpyDeviceToHost,stream1) );
      }

      // Can compute manetisation while grids are copying
      if (isweep%mag_output_int==0){
        compute_magnetisation_gpu<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(L, nreplicas, d_ising_grids, d_magnetisation);    
        gpuErrchk( cudaMemcpyAsync(magnetisation,d_magnetisation,nreplicas*sizeof(float),cudaMemcpyDeviceToHost, stream2) );
      } 

      // MC Sweep - GPU
      gpuErrchk( cudaStreamSynchronize(stream1) ); // Make sure copy completed before making changes

      if (gpu_method==0){
        mc_sweep_gpu<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(L,d_state,nreplicas,d_ising_grids,d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
      } else if (gpu_method==1){
          size_t shmem_size = L*L*threadsPerBlock*sizeof(uint8_t)/8; // number of bytes needed to store grid as bits
          mc_sweep_gpu_bitrep<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,nreplicas,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
      } else if (gpu_method==2){
          size_t shmem_size = L*L*threadsPerBlock*sizeof(uint8_t)/8; // number of bytes needed to store grid as bits
          if (threadsPerBlock==32){
            mc_sweep_gpu_bitmap32<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,nreplicas,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
          } else if (threadsPerBlock==64){
            mc_sweep_gpu_bitmap64<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,nreplicas,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
          } else {
            printf("Invalid threadsPerBlock for gpu_method=2\n");
            exit(EXIT_FAILURE);
          } 
      } else {
        printf("Unknown gpu_method in ising.cu\n");
        exit(EXIT_FAILURE);
      }
      
      // Writing of the grids can be happening on the host while the device runs the mc_sweep kernel
      if (isweep%grid_output_int==0){
        write_ising_grids(L, nreplicas, ising_grids, isweep);  
      }

      // Write and report magnetisation - can also be happening while the device runs the mc_sweep kernel
      if (isweep%mag_output_int==0){
        gpuErrchk( cudaStreamSynchronize(stream2) );  // Wait for copy to complete
        //for (igrid=0;igrid<nreplicas;igrid++){
        //  printf("    %4d     %10d      %8.6f\n",igrid, isweep, magnetisation[igrid]);
        //}
        if ( itask == 0 ) { // Report how many samples have nucleated.
          int nnuc = 0;
          for (igrid=0;igrid<nreplicas;igrid++){
            if ( magnetisation[igrid] > up_threshold ) nnuc++;
          }
          printf("%10d  %12.6f\n",isweep, (double)nnuc/(double)nreplicas);
          if (nnuc==nreplicas) break; // Stop if everyone has nucleated
        } else if ( itask == 1 ){

            // Statistics on fate of trajectories
            int nA=0, nB=0;
            for (igrid=0;igrid<nreplicas;igrid++){
              if (grid_fate[igrid]==0 ) {
                nA++;
              } else if (grid_fate[igrid]==1 ) {
                nB++;
              } else {
                if ( magnetisation[igrid] > up_threshold ){
                  grid_fate[igrid] = 1;
                  nB++;
                } else if (magnetisation[igrid] < dn_threshold ){
                  grid_fate[igrid] = 0;
                  nA++;
                }
              } // fate
            } //grids

            // Monitor progress
            pB = (double)nB/(double)(nA+nB);
            printf("\r Sweep : %10d, Reached m = %6.2f : %4d , Reached m = %6.2f : %4d , Unresolved : %4d, pB = %10.6f",
            isweep, dn_threshold, nA, up_threshold, nB, nreplicas-nA-nB,pB);
            fflush(stdout);
            if (nA + nB == nreplicas) break; // all fates resolved
        
        } // task 
      }

      // Increment isweep
      isweep += sweeps_per_call;

      // Make sure all kernels updating the grids are finished before starting magnetisation calc
      gpuErrchk( cudaStreamSynchronize(stream1) );
      gpuErrchk( cudaPeekAtLastError() );

    }

    // Ensure all threads finished before stopping timer
    gpuErrchk( cudaDeviceSynchronize() )

    t2 = clock();

    printf("\n# Time taken on GPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);
    if (itask==1) { printf("pB estimate : %10.6f\n",pB); }

    // Destroy streams
    gpuErrchk( cudaStreamDestroy(stream1) );
    gpuErrchk( cudaStreamDestroy(stream2) );


    // Free magnetisation arrays
    free(magnetisation);
    gpuErrchk( cudaFree(d_magnetisation) );

  }


/*=================================================
    Tidy up memory used in both GPU and CPU paths
  =================================================*/ 
  free(ising_grids);
  if (run_gpu==true) {
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_neighbour_list) );
  }

  return EXIT_SUCCESS;

}
