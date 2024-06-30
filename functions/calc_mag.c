#include "calc_mag.h" // Includes all function definitions

#define MOD(a,b) ((((a)%(b))+(b))%(b)) // Custom definition of modulus function so that it works correctly for negative values.

int calculate_magnetization(int L, int *grid) {

    int i=0;
    int magnetization=0;

    //printf("Begin Ncon loop\n");
    for (i=0; i<L*L; i++) {
        magnetization = magnetization + grid[i];
    }

    return magnetization;
}
