#include "io.h"
#include <hdf5.h>

void read_input_grid(int L, int ngrids, int *ising_grids){

    // converts [0,1] to [-1,1]
    const int blookup[2] = {-1, 1};

    // Set filename
    char filename[14];
    sprintf(filename, "gridinput.bin");

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

int write_ising_grids(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size, char *cv, double dn_thr, double up_thr){

    // Set filename
    char filename[15];
    sprintf(filename, "gridstates.bin");
    //printf("%s\n",filename);

    // open file
    FILE *ptr = fopen(filename,"ab");
    if (ptr==NULL){
        FILE *ptr = fopen(filename,"wb"); // open for write if not available for append 
        if (ptr==NULL){
            fprintf(stderr,"Error opening %s for write!\n",filename);
            exit(EXIT_FAILURE);
        }   
    }

    // file header - size of grid, number of grids and current sweep
    fwrite(&L,sizeof(int),1,ptr);
    fwrite(&ngrids,sizeof(int),1,ptr);
    fwrite(&isweep,sizeof(int),1,ptr);

    // pack everything into as few bits as possible
    int nbytes = L*L*ngrids/8;
    if ( (L*L*ngrids)%8 !=0 ) { nbytes++; }
    char *bitgrids = (char *)malloc(nbytes);
    if (bitgrids==NULL){
        fprintf(stderr,"Error allocating output buffer!");
        exit(EXIT_FAILURE);
    }

    // Set zero 
    memset(bitgrids, 0U, nbytes);
    
    uint8_t one = 1;

    int ibit=0, ibyte=0, iint;
    for (iint=0;iint<L*L*ngrids;iint++){ //loop over grid x squares per grid

        if ( ising_grids[iint] == 1 ) {
            bitgrids[ibyte] |= one << ibit;
            //printf("Set bit %d of byte %d\n", ibit, ibyte);
        }

        ibit++;
        if (ibit==8) {
            ibit=0;
            ibyte++;
        }

    }

    // write to file
    fwrite(bitgrids,sizeof(char),nbytes,ptr);

    // Release memory
    free(bitgrids);

    // close file
    fclose(ptr);

    return 0; //Success
    
}

int create_ising_grids_hdf5(int L, int ngrids, int tot_nsweeps, double h, double beta) {
    const char *filename = "gridstates.hdf5";

    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: failed to create HDF5 file '%s'\n", filename);
        return -1;
    }

    /* Create a scalar dataspace for each header value */
    hid_t sid = H5Screate(H5S_SCALAR);
    if (sid < 0) {
        fprintf(stderr, "Error: failed to create HDF5 scalar dataspace\n");
        H5Fclose(file_id);
        return -1;
    }

    /* Write integer scalars */
    hid_t did = H5Dcreate2(file_id, "L", H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did < 0) { fprintf(stderr, "Error: failed to create dataset 'L'\n"); H5Sclose(sid); H5Fclose(file_id); return -1; }
    if (H5Dwrite(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &L) < 0) { fprintf(stderr, "Error: failed to write dataset 'L'\n"); H5Dclose(did); H5Sclose(sid); H5Fclose(file_id); return -1; }
    H5Dclose(did);

    did = H5Dcreate2(file_id, "ngrids", H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did < 0) { fprintf(stderr, "Error: failed to create dataset 'ngrids'\n"); H5Sclose(sid); H5Fclose(file_id); return -1; }
    if (H5Dwrite(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ngrids) < 0) { fprintf(stderr, "Error: failed to write dataset 'ngrids'\n"); H5Dclose(did); H5Sclose(sid); H5Fclose(file_id); return -1; }
    H5Dclose(did);

    did = H5Dcreate2(file_id, "tot_nsweeps", H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did < 0) { fprintf(stderr, "Error: failed to create dataset 'tot_nsweeps'\n"); H5Sclose(sid); H5Fclose(file_id); return -1; }
    if (H5Dwrite(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tot_nsweeps) < 0) { fprintf(stderr, "Error: failed to write dataset 'tot_nsweeps'\n"); H5Dclose(did); H5Sclose(sid); H5Fclose(file_id); return -1; }
    H5Dclose(did);

    /* Write double scalars */
    hid_t did_d = H5Dcreate2(file_id, "h", H5T_NATIVE_DOUBLE, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did_d < 0) { fprintf(stderr, "Error: failed to create dataset 'h'\n"); H5Sclose(sid); H5Fclose(file_id); return -1; }
    if (H5Dwrite(did_d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &h) < 0) { fprintf(stderr, "Error: failed to write dataset 'h'\n"); H5Dclose(did_d); H5Sclose(sid); H5Fclose(file_id); return -1; }
    H5Dclose(did_d);

    did_d = H5Dcreate2(file_id, "beta", H5T_NATIVE_DOUBLE, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did_d < 0) { fprintf(stderr, "Error: failed to create dataset 'beta'\n"); H5Sclose(sid); H5Fclose(file_id); return -1; }
    if (H5Dwrite(did_d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &beta) < 0) { fprintf(stderr, "Error: failed to write dataset 'beta'\n"); H5Dclose(did_d); H5Sclose(sid); H5Fclose(file_id); return -1; }
    H5Dclose(did_d);

    /* Create and initialize total_saved_grids header to 0 */
    hid_t did_tot = H5Dcreate2(file_id, "total_saved_grids", H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did_tot < 0) {
        fprintf(stderr, "Warning: failed to create dataset 'total_saved_grids'\n");
    } else {
        int total_grids = 0;
        if (H5Dwrite(did_tot, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &total_grids) < 0) {
            fprintf(stderr, "Warning: failed to initialize 'total_saved_grids'\n");
        }
        H5Dclose(did_tot);
    }

    /* Cleanup */
    H5Sclose(sid);
    H5Fclose(file_id);

    return 0;
}

int write_ising_grids_hdf5(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size, char *cv, double dn_thr, double up_thr) {
    const char *filename = "gridstates.hdf5";
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: failed to open HDF5 file '%s' for writing\n", filename);
        return -1;
    }

    char groupname[64];
    snprintf(groupname, sizeof(groupname), "/sweep_%d", isweep);

    hid_t grp = H5Gcreate2(file_id, groupname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (grp < 0) {
        fprintf(stderr, "Error: failed to create group '%s' in HDF5 file\n", groupname);
        H5Fclose(file_id);
        return -1;
    }

     /* Count successfully written grids and update total_saved_grids accordingly.
         Do not create the dataset here; assume it exists (created by create_ising_grids_hdf5).
     */
     int counter = 0;

    /* Prepare string datatype for attributes (variable-length C strings) */
    hid_t str_tid = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_tid, H5T_VARIABLE);
    hid_t attr_sid = H5Screate(H5S_SCALAR);

    /* For HDF5 storage we will pack 8 Ising spins per byte: +1 -> 1, -1 -> 0.
       Each grid will be stored as a 1D dataset of unsigned bytes. */
    for (int g = 0; g < ngrids; ++g) {
        char dsetname[64];
        snprintf(dsetname, sizeof(dsetname), "grid_%d", g);

        /* number of bits and bytes needed for this grid */
        size_t nbits = (size_t)L * (size_t)L;
        size_t nbytes = nbits / 8;

        hsize_t bdims[1]; bdims[0] = nbytes;
        hid_t dspace = H5Screate_simple(1, bdims, NULL);
        if (dspace < 0) {
            fprintf(stderr, "Error: failed to create dataspace for '%s'\n", dsetname);
            continue;
        }

        hid_t dset = H5Dcreate2(grp, dsetname, H5T_NATIVE_UCHAR, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "Error: failed to create dataset '%s'\n", dsetname);
            H5Sclose(dspace);
            continue;
        }

        /* Allocate buffer for packed bytes */
        unsigned char *buf = (unsigned char *)malloc(nbytes);
        if (buf == NULL) {
            fprintf(stderr, "Error allocating output buffer for '%s'\n", dsetname);
            H5Dclose(dset);
            H5Sclose(dspace);
            continue;
        }
        memset(buf, 0, nbytes);

        /* Pack bits: LSB first (bit 0 corresponds to first site). */
        size_t ibit = 0, ibyte = 0;
        int *grid_ptr = &ising_grids[g * L * L];
        for (size_t i = 0; i < nbits; ++i) {
            if (grid_ptr[i] == 1) {
                buf[ibyte] |= (unsigned char)(1U << ibit);
            }
            ibit++;
            if (ibit == 8) { ibit = 0; ibyte++; }
        }

        /* Write packed bytes */
        if (H5Dwrite(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0) {
            fprintf(stderr, "Error: failed to write dataset '%s'\n", dsetname);
        } else {
            counter++;
        }

        free(buf);

        /* Prepare attribute strings */
        const char *mstr = "null";
        char mval[64];
        if (magnetisation != NULL) {
            snprintf(mval, sizeof(mval), "%g", (double)magnetisation[g]);
            mstr = mval;
        }
        hid_t attr = H5Acreate2(dset, "magnetisation", str_tid, attr_sid, H5P_DEFAULT, H5P_DEFAULT);
        if (attr >= 0) { H5Awrite(attr, str_tid, &mstr); H5Aclose(attr); }

        const char *lstr = "null";
        char lval[64];
        if (lclus_size != NULL) {
            snprintf(lval, sizeof(lval), "%g", (double)lclus_size[g]);
            lstr = lval;
        }
        attr = H5Acreate2(dset, "lclus_size", str_tid, attr_sid, H5P_DEFAULT, H5P_DEFAULT);
        if (attr >= 0) { H5Awrite(attr, str_tid, &lstr); H5Aclose(attr); }

        /* Dummy entries for committor and committor_error, initialized to "null" */
        const char *cnull = "null";
        attr = H5Acreate2(dset, "committor", str_tid, attr_sid, H5P_DEFAULT, H5P_DEFAULT);
        if (attr >= 0) { H5Awrite(attr, str_tid, &cnull); H5Aclose(attr); }

        attr = H5Acreate2(dset, "committor_error", str_tid, attr_sid, H5P_DEFAULT, H5P_DEFAULT);
        if (attr >= 0) { H5Awrite(attr, str_tid, &cnull); H5Aclose(attr); }

        H5Dclose(dset);
        H5Sclose(dspace);
    }

    H5Sclose(attr_sid);
    H5Tclose(str_tid);
    H5Gclose(grp);

    /* Update total_saved_grids without error checks; if no grids written delete the sweep group */
    if (counter > 0) {
        hid_t tot_did = H5Dopen2(file_id, "total_saved_grids", H5P_DEFAULT);
        int total_saved = 0;
        H5Dread(tot_did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &total_saved);
        total_saved += counter;
        H5Dwrite(tot_did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &total_saved);
        H5Dclose(tot_did);
    } else {
        H5Ldelete(file_id, groupname, H5P_DEFAULT);
    }

    H5Fclose(file_id);

    return 0;
}
