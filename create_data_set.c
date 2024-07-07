#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include "functions/read_input_variables.h" 
#include "functions/comparison.h" 

#define MOD(a,b) ((((a)%(b))+(b))%(b)) // Custom definition so that mod works correctly with negative numbers

int main (int argc, char *argv[]) {
    
    // Setup for timing code
    clock_t start, end;
    double execution_time;
    start = clock();

    // Initialise pseudo-random number generation
    srand(time(NULL));

    // Process commandline input
    if (argc != 4) {
        printf("Usage : samples_per_mag_size min_mag max_mag\n");
        printf("Set either min_mag or max_mag to -1 to use default value.\n");
        exit(EXIT_FAILURE);
    }
    
    // Define and read input variables
    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method;
    double beta, h;
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    int samples;
    int min_mag = 10;
    int max_mag = 500;

    samples = atoi(argv[1]); // Number of samples to be chosen
    if (atoi(argv[2]) != -1) {
        min_mag = atoi(argv[2]);
    }
    if (atoi(argv[2]) != -1) {
        max_mag = atoi(argv[3]);
    }

    // Set filenames
    const char *index_filename = "index.bin";
    const char *committor_filename = "committor_index.bin";

    // Open file to read
    FILE *index_file = fopen(index_filename,"rb");
    if (index_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", index_filename);
        exit(EXIT_FAILURE);
    }

    // Remove committor file
    remove(committor_filename);
    // Create file to write
    FILE *committor_file = fopen(committor_filename,"wb");
    if (committor_file==NULL){
        fprintf(stderr,"Error opening %s for write!\n",committor_filename);
        exit(EXIT_FAILURE);
    }
    
    // Create loop variables
    int i = 0, j = 0, k = 0;
    int islice, igrid;
    int mag = 0;

    // Create arrays for storing for ngrid, islice, cluster and spare for committor        
    int *store_ngrid = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_ngrid==NULL){fprintf(stderr,"Error allocating memory for store_ngrid array!\n"); exit(EXIT_FAILURE);} 
    int *store_slice = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_slice==NULL){fprintf(stderr,"Error allocating memory for store_slice array!\n"); exit(EXIT_FAILURE);} 
    int *store_mag = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_mag==NULL){fprintf(stderr,"Error allocating memory for store_mag array!\n"); exit(EXIT_FAILURE);} 
    double *store_committor = (double *)malloc(nreplicas*nsweeps/100*sizeof(double));
    if (store_committor==NULL){fprintf(stderr,"Error allocating memory for store_committor array!\n"); exit(EXIT_FAILURE);} 

    // Read and sotre data from index file
    // Loops over slices
    for (islice=0;islice<nsweeps/100;islice++) {
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            fread(&store_slice[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_ngrid[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_mag[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_committor[igrid+nreplicas*islice], sizeof(double), 1, index_file);
            fread(&store_committor[igrid+nreplicas*islice], sizeof(double), 1, index_file); // Dummy read that reads the standard deviation value in, which at this stage is also -1
        }
    }

    // Sort the loaded arrays based on the cluster size
    int **p_store_mag = malloc(nreplicas*nsweeps/100*sizeof(long));
    int ta, tb, tc, td;

    // create array of pointers to store_cluster
    for (i = 0; i < nreplicas*nsweeps/100; i++) {
        p_store_mag[i] = &store_mag[i];
    }

    // sort array of pointers
    qsort(p_store_mag, nreplicas*nsweeps/100, sizeof(long), compare);
    
    // reorder loaded arrays according to the array of pointers
    for(i=0;i<nreplicas*nsweeps/100;i++){
        if(i != p_store_mag[i]-store_mag){
            ta = store_ngrid[i];
            tb = store_slice[i];
            tc = store_mag[i];
            td = store_committor[i];
            k = i;
            while(i != (j = p_store_mag[k]-store_mag)){
                store_ngrid[k] = store_ngrid[j];
                store_slice[k] = store_slice[j];
                store_mag[k] = store_mag[j];
                store_committor[k] = store_committor[k];
                p_store_mag[k] = &store_mag[k];
                k = j;
            }
            store_ngrid[k] = ta;
            store_slice[k] = tb;
            store_mag[k] = tc;
            store_committor[k] = td;
            p_store_mag[k] = &store_mag[k];
        }
    }

    // Create array for storing starting index of each cluster size and how many of a given cluster exist
    int *store_mag_index = (int *)malloc(L*L*2*2*sizeof(int));
    if (store_mag_index==NULL){fprintf(stderr,"Error allocating memory for store_mag_index array!\n"); exit(EXIT_FAILURE);}
    for (i=0;i<L*L*2*2;i++) {store_mag_index[i]=0;}

    int mag_check = 0;
    mag = 0;
    for (i=0;i<nreplicas*nsweeps/100;i++) {
        mag = store_mag[i]+L*L;
        if ( mag > mag_check) {
            mag_check = mag;
            store_mag_index[(mag)*2] = i;
        }
        store_mag_index[(mag)*2+1] = store_mag_index[(mag)*2+1]+1;
    }

    int random_selection = 0; // Used to randomly select grid from index.bin file
    int selected_mag = 0;
    int samples_tmp = samples;

    // Create array for storing random selections
    int *rand_array = (int *)malloc(samples*sizeof(int));
    if (rand_array==NULL){fprintf(stderr,"Error allocating memory for rand_array array!\n"); exit(EXIT_FAILURE);}
    for (i=0;i<samples;i++) {rand_array[i]=-1;}

    int counter = 0;
    int in, im, rn, rm;
    int rand_loops = 0;
    int full_iterations = 0;
    int partial_iterations = 0;

    full_iterations = (int)samples/(L*L*2);
    partial_iterations = samples % (L*L*2);

    for (i=0;i<full_iterations;i++) {
        im = 0;
        for (in = 0; in < max_mag-min_mag && im < max_mag-min_mag; ++in) {
            rn = max_mag-min_mag - in;
            rm = max_mag-min_mag - im;
            if (rand() % rn < rm && store_mag_index[(in + min_mag)*2+1] > i) {
                rand_array[i*(L*L*2)+im++] = in + min_mag;
            }
        }
    }
    
    im = 0;
    for (in = 0; in < max_mag-min_mag && im < partial_iterations; ++in) {
        rn = max_mag-min_mag - in;
        rm = partial_iterations - im;
        if (rand() % rn < rm && store_mag_index[(in + min_mag)*2+1] > i) {
            rand_array[i*(L*L*2)+im++] = in + min_mag;
        }
    }

   int **p_rand_array = malloc(samples*sizeof(long));

    // create array of pointers to store_cluster
    for (i = 0; i < samples; i++) {
        p_rand_array[i] = &rand_array[i];
    }

    // sort array of pointers
    qsort(p_rand_array, samples, sizeof(long), compare);

    // reorder loaded arrays according to the array of pointers
    for(i=0;i<samples;i++){
        if(i != p_rand_array[i]-rand_array){
            ta = rand_array[i];
            k = i;
            while(i != (j = p_rand_array[k]-rand_array)){
                rand_array[k] = rand_array[j];
                p_rand_array[k] = &rand_array[k];
                k = j;
            }
            rand_array[k] = ta; 
            p_rand_array[k] = &rand_array[k];
        }
    }

    i = 1; j = 0;
    int out_num = 0;
    int rand_array_start = 1;
    while(i < samples){
        if(rand_array[i] != out_num && rand_array[i] > -1){
            out_num = rand_array[i];
            j += 1;
        }
        if(rand_array[i] < 0) {
            rand_array_start++;
        }
        i++;
    }

    int unique_rand = j;
    int *unique_rand_array = (int *)malloc(unique_rand*2*sizeof(int));
    if (unique_rand_array==NULL){fprintf(stderr,"Error allocating memory for unique_rand_array array!\n"); exit(EXIT_FAILURE);}
    for (i = 0; i < unique_rand*2; i++) {unique_rand_array[i] = 0;}

    i = rand_array_start+1; j = 1; k = 1;
    unique_rand_array[0] = rand_array[rand_array_start];

    while(i<samples){
        if(rand_array[i] != unique_rand_array[j*2]){
            unique_rand_array[j*2] = rand_array[i];
            unique_rand_array[(j-1)*2+1] = k;
            k = 0;
            j++;
        }
        i++; k++;
    }
    unique_rand_array[(j-1)*2+1] = k;

    printf("Total available samples: %d\n", samples-rand_array_start);

    int *rand_array_sub = (int *)malloc((full_iterations+1)*sizeof(int));
    if (rand_array_sub==NULL){fprintf(stderr,"Error allocating memory for rand_array_sub array!\n"); exit(EXIT_FAILURE);}
    for (i = 0; i < full_iterations+1; i++) {rand_array_sub[i] = 0;}

    for (i = 0; i < unique_rand; i++) {
        im = 0;
        for (in = 0; in < store_mag_index[unique_rand_array[i*2]*2+1] && im < unique_rand_array[i*2+1]; ++in) {
            rn = store_mag_index[unique_rand_array[i*2]*2+1] - in;
            rm = unique_rand_array[i*2+1] - im;
            if (rand() % rn < rm) {
                rand_array_sub[im++] = in;
            }
        }
        
        for (j = 0; j < unique_rand_array[i*2+1]; j++) {
            random_selection = rand_array_sub[j]+store_mag_index[unique_rand_array[i*2]*2];
            selected_mag = store_mag[random_selection];
            fwrite(&store_slice[random_selection], sizeof(int), 1, committor_file);
            fwrite(&store_ngrid[random_selection], sizeof(int), 1, committor_file);
            fwrite(&selected_mag, sizeof(int), 1, committor_file);
            fwrite(&store_committor[random_selection], sizeof(double), 1, committor_file);
            fwrite(&store_committor[random_selection], sizeof(double), 1, committor_file); // Write to create space for standard deviation
            counter += 1;
        }
        //printf("\rPercentage of samples selected: %d%%", (int)(100.0*(double)counter/(double)(samples-rand_array_start))); // Print progress
        fflush(stdout);
    } 

    for (i = 0; i < unique_rand; i++) {
        printf("%d %d %d\n", unique_rand_array[i*2], store_mag_index[unique_rand_array[i*2]*2], store_mag_index[unique_rand_array[i*2]*2+1]);
    }

    printf("\n"); // Newline

    free(store_ngrid); free(store_slice); free(store_mag); free(store_mag_index); free(store_committor); free(rand_array); free(rand_array_sub); free(unique_rand_array);
    free(p_store_mag); free(p_rand_array);
    fclose(index_file); fclose(committor_file);

    // Print time taken for program to execute
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken: %.2f seconds \n", execution_time);
    
    return EXIT_SUCCESS;
}  
