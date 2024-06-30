#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
// Includes function definitions
#include "functions/read_input_variables.h" 
#include "functions/read_input_grid.h"
#include "functions/calc_mag.h" 

#define MOD(a,b) ((((a)%(b))+(b))%(b))

// Main function to find cluster size and write it to file

int main() {

    // Setup for timing code
    clock_t start, end;
    double execution_time;
    start = clock();

    // Define and read input variables
    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method;
    double beta, h;
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    // Define maximum connetions per grid point. 4 in this case since 2D nearest neighbour Ising model is being used
    int Nvert=L*L;

    // Set filenames
    const char *read_file_name = "gridstates.bin";
    const char *write_file_name = "index.bin";
    
    // open file to read grid
    FILE *read_file = fopen(read_file_name, "rb");
    if (read_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", read_file_name);
        exit(EXIT_FAILURE);
    }

    // Delete previous index file
    remove("index.bin");

    // create file to write indexing
    FILE *write_file = fopen(write_file_name, "wb");
    if (write_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", write_file_name);
        exit(EXIT_FAILURE);
    }

    // Host copy of Ising grid configurations
    int *ising_grids = (int *)malloc(L*L*sizeof(int));
    if (ising_grids==NULL){
        fprintf(stderr,"Error allocating memory for Ising grids!\n");
        exit(EXIT_FAILURE);
    } 

    // Define loop indices
    int islice,igrid;

    // Create output array for ngrid, islice, cluster and spare for commitor        
    int *output_ngrid = (int *)malloc(nreplicas*sizeof(int));
    if (output_ngrid==NULL){fprintf(stderr,"Error allocating memory for output_ngrid array!\n"); exit(EXIT_FAILURE);} 
    int *output_slice = (int *)malloc(nreplicas*sizeof(int));
    if (output_slice==NULL){fprintf(stderr,"Error allocating memory for output_slice array!\n"); exit(EXIT_FAILURE);} 
    int *output_mag = (int *)malloc(nreplicas*sizeof(int));
    if (output_mag==NULL){fprintf(stderr,"Error allocating memory for output_cluster array!\n"); exit(EXIT_FAILURE);} 
    double *output_commitor = (double *)malloc(nreplicas*sizeof(double));
    if (output_commitor==NULL){fprintf(stderr,"Error allocating memory for output_commitor array!\n"); exit(EXIT_FAILURE);} 

    // Allocate space to read a single grid as bits
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes++; }
    char *bitgrid = (char *)malloc(nbytes);
    if (bitgrid==NULL){
        fprintf(stderr,"Error allocating input buffer for ising grid!");
        exit(EXIT_FAILURE);
    }

    /*--------------------------------------------/
    / Allocate memory to hold graph connectivity  /
    /--------------------------------------------*/
    int temp_mag = 0;

    // Main loop which finds magnetization and writes it to file
    // Loops over slices i.e. sweep snapshots

    for (islice=0;islice<nsweeps/100;islice++) {
        printf("\rPercentage of magnetizations calculated: %d%%", (int)((double)(islice+1)/(double)(nsweeps/100)*100)); // Print progress
        fflush(stdout);
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            read_input_grid(read_file, bitgrid, L, ising_grids, nreplicas, islice, igrid);
            // Saves grid number, slice, cluster size and spare data entry for commitor
            temp_mag = calculate_magnetization(L, ising_grids);
            output_ngrid[igrid] = igrid;
            output_slice[igrid] = islice*100;
            output_mag[igrid] = temp_mag;
            output_commitor[igrid] = (double)-1;
        } // igrid
        for (igrid=0;igrid<nreplicas;igrid++) {
            fwrite(&output_slice[igrid], sizeof(int), 1, write_file);
            fwrite(&output_ngrid[igrid], sizeof(int), 1, write_file);
            fwrite(&output_mag[igrid], sizeof(int), 1, write_file);
            fwrite(&output_commitor[igrid], sizeof(double), 1, write_file);
            fwrite(&output_commitor[igrid], sizeof(double), 1, write_file); // Additional write to store standard deviation on commitor
        }
        
        
    } // isweep

    // Free memory
    free(bitgrid); free(output_ngrid); free(output_slice); free(output_mag); free(output_commitor); free(ising_grids);

    // Close files
    fclose(write_file); fclose(read_file);
    
    // New line
    printf("\n");

    printf("Magnetization calculation successfully completed. \n");

    // Print time taken for program to execute
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken: %.2f seconds \n", execution_time);

    return EXIT_SUCCESS;
}


/*
 *
 * rowcol(i)
 *  row = i/L
 *  col = i%L
 *
 * index(row,col)
 *  index = col + row*L
*/
