#include "read_input_variables.h"

// Function which takes variable storage and a file name, then opens the file with given file name and reads input variables into provided storage. Closes file once finished.
void read_input_variables(int *L, int *nreplicas, int *nsweeps, int *mag_output_int, int *grid_output_int, int *threadsPerBlock, int *gpu_device, int *gpu_method, double *beta, double *h) {

    // Checks that input variable file exists
    const char *input_variables_name = "input_variables.bin";
    FILE *input_variables = fopen(input_variables_name,"rb");
    if (input_variables==NULL){
        fprintf(stderr,"Error opening %s for read! File most likely does not exist.\n",input_variables_name);
        exit(EXIT_FAILURE);
    }

    // Reads input variable file data into appropriate variable storage
    fread(L, sizeof(int), 1, input_variables);
    fread(nreplicas, sizeof(int), 1, input_variables);
    fread(nsweeps, sizeof(int), 1, input_variables);
    fread(mag_output_int, sizeof(int), 1, input_variables);
    fread(grid_output_int, sizeof(int), 1, input_variables);
    fread(threadsPerBlock, sizeof(int), 1, input_variables);
    fread(gpu_device, sizeof(int), 1, input_variables);
    fread(gpu_method, sizeof(int), 1, input_variables);
    fread(beta, sizeof(double), 1, input_variables);
    fread(h, sizeof(double), 1, input_variables);

    // Close input variable file
    fclose(input_variables);
}