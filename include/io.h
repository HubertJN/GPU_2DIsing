#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <hdf5.h>

void write_ising_grids(int L, int ngrids, int *ising_grids, int isweep, float h, float beta);
void read_input_grid(int L, int ngrids, int *ising_grids);