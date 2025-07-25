#include "io.h"


void read_input_grid(int L, int ngrids, int *ising_grids){

    // converts [0,1] to [-1,1]
    const int blookup[2] = {-1, 1};

    // Set filename
    char filename[18];
    sprintf(filename, "out/gridinput.bin");

    uint32_t one = 1U;   

    // open file
    FILE *ptr = fopen(filename, "rb");
    if (ptr==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename);
        exit(EXIT_FAILURE);
    }

    // read header specifying size of grid
    int Lcheck;
    fread(&Lcheck, sizeof(int), 1, ptr);
    if (Lcheck!=L) {
        fprintf(stderr, "Error - size of grid in input file does not match L!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate space to read a single grid as bits
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes++; }
    char *bitgrid = (char *)malloc(nbytes);
    if (bitgrid==NULL){
        fprintf(stderr,"Error allocating input buffer!");
        exit(EXIT_FAILURE);
    }

    // Read the grid
    fread(bitgrid, sizeof(char), nbytes, ptr);  

    // Loop over grid points
    int ibit=0, ibyte=0;
    int isite=0, igrid;

    //printf("nbytes = %d\n",nbytes);
    for (ibyte=0;ibyte<nbytes;ibyte++){
        for (ibit=0;ibit<8;ibit++){
            //printf(" %2d ",blookup[(bitgrid[ibyte] >> ibit) & one]);
            // Read into every copy of the grid
            for (igrid=0;igrid<ngrids;igrid++){
                ising_grids[L*L*igrid+isite] = blookup[(bitgrid[ibyte] >> ibit) & one];
            }
            isite++;
            //if (isite%L==0) {printf("\n");}
        }
        if (isite>L*L) break;
    }

    free(bitgrid);  // free input buffer
    fclose(ptr);    // close input file

    fprintf(stderr, "Read initial configuration of all grids from gridinput.bin\n");

}

void write_ising_grids(int L, int ngrids, int *ising_grids, int isweep, float h, float beta) {
    const char *filename = "out/ising_data.h5";

    // Calculate packed buffer size
    int nbits = L * L * ngrids;
    int nbytes = nbits / 8;
    if (nbits % 8 != 0) nbytes++;

    // Allocate bit buffer
    unsigned char *bitgrids = (unsigned char *)malloc(nbytes);
    if (!bitgrids) {
        fprintf(stderr, "Error allocating bit buffer\n");
        exit(EXIT_FAILURE);
    }
    memset(bitgrids, 0, nbytes);

    // Pack bits
    uint8_t one = 1;
    int ibit = 0, ibyte = 0;
    for (int i = 0; i < nbits; i++) {
        if (ising_grids[i] == 1) {
            bitgrids[ibyte] |= one << ibit;
        }
        ibit++;
        if (ibit == 8) {
            ibit = 0;
            ibyte++;
        }
    }

    // Open or create file
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "Error creating file %s\n", filename);
            free(bitgrids);
            exit(EXIT_FAILURE);
        }

        // Write file-level attributes
        hid_t attr_space = H5Screate(H5S_SCALAR);
        hid_t attr;

        attr = H5Acreate(file_id, "L", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &L);
        H5Aclose(attr);

        attr = H5Acreate(file_id, "ngrids", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &ngrids);
        H5Aclose(attr);

        attr = H5Acreate(file_id, "h", H5T_NATIVE_FLOAT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_FLOAT, &h);
        H5Aclose(attr);

        attr = H5Acreate(file_id, "beta", H5T_NATIVE_FLOAT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_FLOAT, &beta);
        H5Aclose(attr);

        H5Sclose(attr_space);
    }

    // Create group for this sweep
    char groupname[32];
    snprintf(groupname, sizeof(groupname), "/sweep_%06d", isweep);

    if (H5Lexists(file_id, groupname, H5P_DEFAULT)) {
        fprintf(stderr, "Warning: Group %s already exists. Overwriting dataset.\n", groupname);
    }

    hid_t group_id = H5Gcreate(file_id, groupname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) {
        fprintf(stderr, "Error creating group %s\n", groupname);
        free(bitgrids);
        H5Fclose(file_id);
        exit(EXIT_FAILURE);
    }

    // Create dataspace for packed data: 1D array of bytes
    hsize_t dims[1] = { (hsize_t)nbytes };
    hid_t space_id = H5Screate_simple(1, dims, NULL);

    // Create dataset for packed grids
    hid_t dset_id = H5Dcreate(group_id, "packed_grids", H5T_STD_U8LE, space_id,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset_id < 0) {
        fprintf(stderr, "Error creating dataset packed_grids\n");
        free(bitgrids);
        H5Sclose(space_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        exit(EXIT_FAILURE);
    }

    // Write packed data
    herr_t status = H5Dwrite(dset_id, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, bitgrids);
    if (status < 0) {
        fprintf(stderr, "Error writing dataset packed_grids\n");
    }

    // Cleanup
    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Gclose(group_id);
    H5Fclose(file_id);

    free(bitgrids);
}
