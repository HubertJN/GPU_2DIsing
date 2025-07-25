#include "mc_cpu.h"

double Pacc[20];         // Cache of acceptance probabilities


// populate acceptance probabilities
void preComputeProbs_cpu(double beta, double h) {

  // Pre-compute acceptance probabilities for all possibilities
  // s  : nsum               : index 5*(s+1) + nsum + 4
  // +1 : -4 = -1 -1 -1 -1   : 10
  // +1 : -2 = -1 -1 -1 +1   : 12
  // +1 :  0 = -1 -1 +1 +1   : 14
  // +1 : +2 = -1 +1 +1 +1   : 16
  // +1 : +4 = +1 +1 +1 +1   : 18
  // -1 : -4 = -1 -1 -1 -1   : 0
  // -1 : -2 = -1 -1 -1 +1   : 2
  // -1 :  0 = -1 -1 +1 +1   : 4
  // -1 : +2 = -1 +1 +1 +1   : 6
  // -1 : +4 = +1 +1 +1 +1   : 8

  int s, nsum, index;
  for (s=-1;s<2;s=s+2){
    for (nsum=-4;nsum<5;nsum=nsum+2){
      index = 5*(s+1) + nsum + 4;
      Pacc[index] = 2.0*(double)s*((double)nsum+h);
      Pacc[index] = exp(-beta*Pacc[index]);
    }
  }

}     

// sweep on the cpu
void mc_sweep_cpu(int L, int *ising_grids, int grid_index, double beta, double h, int nsweeps) {

  // Pointer to the current Ising grid
  int *loc_grid = &ising_grids[grid_index*L*L];

  int imove, row, col;

  for (imove=0;imove<L*L*nsweeps;imove++){

    // pick random spin
    row = floor(L*genrand_real3());  // RNG cannot generate 1.0 so safe
    col = floor(L*genrand_real3()); 

    // find neighbours
    int my_idx = L*row+col;
    int up_idx = L*((row+L-1)%L) + col;  
    int dn_idx = L*((row+1)%L)   + col;   
    int rt_idx = L*row + (col+1)%L;
    int lt_idx = L*row + (col+L-1)%L;
    
    // energy before flip
    int n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
    //double energy_old = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );

    int index = 5*(loc_grid[my_idx]+1) + n_sum + 4;

    // flip
    loc_grid[my_idx] = -1*loc_grid[my_idx];

    // energy after flip
    //n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
    //double energy_new = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );

    //double delta_energy = energy_new - energy_old;
    //double prob = exp(-beta*delta_energy);

    double xi = genrand_real3();
    //if (xi < prob) {
    if (xi < Pacc[index] ) {
      // accept
      //fprintf(stderr,"Accepted a move\n");
    } else {
      loc_grid[my_idx] = -1*loc_grid[my_idx]; // reject
    }


  } // end for

}

void compute_magnetisation_cpu(int L, int *ising_grids, int grid_index, double *magnetisation){

  // Pointer to the current Ising grid
  int *loc_grid = &ising_grids[grid_index*L*L];

    double m = 0.0f;

    int i;
    for (i=0;i<L*L;i++) { m += (double)loc_grid[i]; }
    magnetisation[grid_index] = m/(double)(L*L);

  return;

}
