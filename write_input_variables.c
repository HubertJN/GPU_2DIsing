#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

int main () {
    /*
    Input file to read variables from

    L               : size of grid (LxL grid)
    nsweeps         : maximum number of MC sweeps (one sweep is L^2 attempted flips)  
    nreplicas       : number of independent replicas to run in parallel on the GPU  
    mag_output_int  : sweeps between checking magnetisation against nucleation/committor thresholds  
    grid_output_int : sweeps between binary dumps of simulation grids to gridstates.bin 
    threadsPerBlock : number of threads/replicas per GPU thread block, 32 is standard  
    gpu_device      : index of GPU device to use, set to zero for single GPU system  
    gpu_method      : 0, 1 or 2 as described below  
    beta            : inverse temperature for simulation  
    h               : magnetic field for simulation  
    itask           : 0 for reversal from spin down configuration, 1 for committor calculation
    */
    /**************Change Variables Here****************/
    int L=64;
    int nreplicas=100;
    int nsweeps=50000;
    int mag_output_int=100;
    int grid_output_int=100;
    int threadsPerBlock=32;
    int gpu_device=0;
    int gpu_method=0;
    double beta=0.54;
    double h=0.07;
    /***************************************************/
    // After variables are changed save file and run write_input_variables.sh (input into cmd: bash write_input_variables.sh)
    /***************************************************/
    // Set filenames
    const char *filename = "input_variables.bin";
    FILE *ptr = fopen(filename,"wb"); // open for write if not available for append 
    if (ptr==NULL){
        fprintf(stderr,"Error opening %s for write!\n",filename);
        exit(EXIT_FAILURE);
    }

    /**************************************************/
    fwrite(&L, sizeof(int), 1, ptr);
    fwrite(&nreplicas, sizeof(int), 1, ptr);
    fwrite(&nsweeps, sizeof(int), 1, ptr);
    fwrite(&mag_output_int, sizeof(int), 1, ptr);
    fwrite(&grid_output_int, sizeof(int), 1, ptr);
    fwrite(&threadsPerBlock, sizeof(int), 1, ptr);
    fwrite(&gpu_device, sizeof(int), 1, ptr);
    fwrite(&gpu_method, sizeof(int), 1, ptr);
    fwrite(&beta, sizeof(double), 1, ptr);
    fwrite(&h, sizeof(double), 1, ptr);

    fclose(ptr);

    return EXIT_SUCCESS;
}
