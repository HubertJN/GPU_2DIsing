#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    // Set filenames
    const char *filename1 = "index.bin";

    // open write cluster file
    FILE *ptr1 = fopen(filename1,"rwb"); // open for write if not available for append 
    if (ptr1==NULL){
        fprintf(stderr,"Error opening %s for write!\n",filename1);
        exit(EXIT_FAILURE);
    }

    // Create array to store index
    int *index = (int *)malloc(3*sizeof(int));
    if (index==NULL){
        fprintf(stderr,"Error allocating memory for index!\n");
        exit(EXIT_FAILURE);
    }

    int tot_sample = 0;
    int within_range = 0;
    double tmp1 = 0.0, tmp2 = 0.0;
    double lower = 0.0, upper = 1.0;
    int min = 64*64, max = 0;

    while (1) {
        fread(index, sizeof(int), 3, ptr1);
        fread(&tmp1, sizeof(double), 1, ptr1);
        fread(&tmp2, sizeof(double), 1, ptr1);
        if ( feof(ptr1) ) { break;}
//        printf("%d %d %d %f %f \n", index[0], index[1], index[2], tmp1, tmp2);
    }

    #define M 10
    #define N 100
    
    int in, im;
    int vektor[10];
    im = 0;
    
    for (in = 0; in < N && im < M; ++in) {
      int rn = N - in;
      int rm = M - im;
      if (rand() % rn < rm)    
        /* Take it */
        vektor[im++] = in + 1; /* +1 since your range begins from 1 */     
    }
    for (in = 0; in<10; in++) {printf("%d ", vektor[in]);}
    printf("\n");
    return(EXIT_SUCCESS);
}
