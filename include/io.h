#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <hdf5.h>

// Function which reads input Ising grid from file (gridinput.bin)
void read_input_grid(int L, int ngrids, int *ising_grids);

// Function to write Ising grids to binary file (gridstates.bin)
int write_ising_grids(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size, char *cv, double dn_thr, double up_thr, char *filename);

// Functions to create HDF5 header and data records for output grids
int create_ising_grids_hdf5(int L, int ngrids, int tot_nsweeps, double h, double beta, int itask, char *filename);
int write_ising_grids_hdf5(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size, char *cv, double dn_thr, double up_thr, char *filename);



