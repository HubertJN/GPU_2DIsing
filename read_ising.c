#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>

int main() {
unsigned char buffer[1000];
FILE *ptr;

ptr = fopen("gridstates.bin","rb");  // r for read, b for binary

fread(buffer,sizeof(buffer),1,ptr); // read 10 bytes to our buffer

for(int i = 0; i<1000; i++)
    printf("%u ", buffer[i]); // prints a series of bytes
printf("\n");
return EXIT_SUCCESS;

}
