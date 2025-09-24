#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Function to compute an estimate of the error bar on the committor given the array of grid fates
double bootstrap_errbar(int ngrids, int m, int *grid_fate_host);