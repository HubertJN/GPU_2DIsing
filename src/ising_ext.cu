// -*- mode: C -*-
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <structmember.h>

extern "C" {
  #include "io.h"
  #include "grid.h"
  #include "bootstrap.h"
}

#include "mc_gpu.h"


bool run_gpu = true;    // Run using GPU
bool run_cpu = false;   // Run using CPU

int idev = -1; // GPU device to use
int gpu_nsms;  // Number of multiprocessors on the GPU


/* Pointer to memory in which we might store copies of all grids 
   generated to pass back to Python. 8 bit integers to save RAM */
int8_t* grid_history = NULL;
int ihist = 0;                 // Current snapshot number

/* Struct which represents an instance of a grid snapshot object */
typedef struct {
    PyObject_HEAD
    int isweep;             // MC sweep at which the grid was captured
    int igrid;              // Which grid in the ensemble is this
    float magnetisation;    // Sum of spins
    float lclus_size;       // Size of largest cluster of spins opposite to initial spin
    float committor;        // Committor value
    PyObject *grid;         // Numpy array that holds the grid of spins
} GridSnapObject;


/* Struct which defines the members of the struct as they'll be seen by Python */
static PyMemberDef GridSnapObject_members[] = {
    {"isweep", T_INT, offsetof(GridSnapObject, isweep), 0, "MC sweep at which the grid was captured"},
    {"igrid", T_INT, offsetof(GridSnapObject, igrid), 0, "Which grid in the ensemble is this"},
    {"magnetisation", T_FLOAT, offsetof(GridSnapObject, magnetisation), 0, "Magnetisation - i.e. sum of spins"},
    {"lclus_size", T_FLOAT, offsetof(GridSnapObject, lclus_size), 0, "Size of largest cluser of spins opposite to initial spin"},
    {"committor", T_FLOAT, offsetof(GridSnapObject, committor), 0, "Committor value if available. -1 if not."},
    //{"grid", T_OBJECT, offsetof(GridSnapObject, grid), 0, "Numpy array that holds the grid of spins"},
    {NULL}  /* Sentinel */
};

/* We don't let Python access the grid via the struct, instead we define get/set functions so we can handle reference counting*/
static PyObject *get_grid(GridSnapObject *self, void *closure) {
    Py_INCREF(self->grid); // Increment reference count before returning
    return self->grid;
}

static int set_grid(GridSnapObject *self, PyObject *value, void *closure) {
    if (!PyArray_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "grid must be a numpy array");
        return -1;
    }
    Py_XDECREF(self->grid); // Decrement old reference count
    self->grid = value;
    Py_INCREF(self->grid); // Increment new reference count
    return 0;
}

static PyGetSetDef GridSnapObject_getset[] = {
    {"grid", (getter)get_grid, (setter)set_grid, "2D numpy array of 8-bit integers", NULL},
    {NULL}  /* Sentinel */
};




/* Constructor for the GridSnapObject type */
static int GridSnapObject_init(GridSnapObject *self, PyObject *args, PyObject *kwds) {
    long int in_isweep, in_igrid;
    PyObject *in_numpy_array = NULL;

    // A format string for parsing arguments: "iiO" means two long integers and one Python object
    if (!PyArg_ParseTuple(args, "iiO", &in_isweep, &in_igrid, &in_numpy_array)) {
        return -1; // Parsing failed
    }

    // Check if the provided object is a NumPy array of the correct type and dimensions
    if (!PyArray_Check(in_numpy_array)) {
        PyErr_SetString(PyExc_TypeError, "Third argument must be a NumPy array.");
        return -1;
    }

    // Check for 8-bit integers (np.int8) and 2 dimensions
    PyArrayObject *arr = (PyArrayObject *)in_numpy_array;
    if (PyArray_TYPE(arr) != NPY_INT8 || PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_TypeError, "NumPy array must be 2D with 8-bit integer elements.");
        return -1;
    }

    // Set the properties of the C struct
    self->isweep = in_isweep;
    self->igrid = in_igrid;
    self->magnetisation = 0.0; // Default value (will always be available)
    self->lclus_size = -1.0;   // Default value (-1 if not available)
    self->committor = -1.0;    // Default value (-1 if not available)

    // Assign the NumPy array and manage its reference count
    Py_INCREF(in_numpy_array); // Increment the reference count for the new object
    self->grid = in_numpy_array;

    return 0; // Success
}

/* Deallocator for the GridSnapObject type */
static void GridSnapObject_dealloc(GridSnapObject *self) {

    // Release the reference to the NumPy grid
    Py_XDECREF(self->grid);

    // Call the base type's deallocator to free the object's memory
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// Forward declaration
static PyObject* GridSnapObject_deepcopy(PyObject* self, PyObject* memo);


// Add __deepcopy__ to GridSnapObjectType
static PyMethodDef GridSnapObject_methods[] = {
  {"__deepcopy__", (PyCFunction)GridSnapObject_deepcopy, METH_VARARGS, "Deep copy a GridSnapObject."},
  {NULL, NULL, 0, NULL}
};

/* Define the Types that GridSnapObjects are an instance of */
PyTypeObject GridSnapObjectType = {
    PyObject_HEAD_INIT(NULL)
  .tp_name = "gasp.GridSnapObject",
  .tp_basicsize = sizeof(GridSnapObject),
  .tp_dealloc = (destructor)GridSnapObject_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_methods = GridSnapObject_methods,
  .tp_members = GridSnapObject_members,
  .tp_getset = GridSnapObject_getset,
  .tp_init = (initproc)GridSnapObject_init,
  .tp_new = PyType_GenericNew,
};


// __deepcopy__ implementation for GridSnapObject
static PyObject* GridSnapObject_deepcopy(PyObject* self, PyObject* memo) {
  GridSnapObject* orig = (GridSnapObject*)self;
  PyObject* grid_copy = PyObject_CallMethod(orig->grid, "copy", NULL);
  if (!grid_copy) return NULL;
  PyObject* args = Py_BuildValue("iiO", orig->isweep, orig->igrid, grid_copy);
  PyObject* new_obj = PyObject_CallObject((PyObject*)&GridSnapObjectType, args);
  Py_DECREF(args);
  Py_DECREF(grid_copy);
  if (!new_obj) return NULL;
  // Copy attributes
  PyObject_SetAttrString(new_obj, "magnetisation", PyFloat_FromDouble((double)orig->magnetisation));
  PyObject_SetAttrString(new_obj, "lclus_size", PyFloat_FromDouble((double)orig->lclus_size));
  PyObject_SetAttrString(new_obj, "committor", PyFloat_FromDouble((double)orig->committor));
  return new_obj;
}



static PyObject* reset_grids_list(PyObject* self, PyObject* args) {
    
  PyObject *module, *new_list;

  // Import the module
  module = PyImport_ImportModule("gasp");
  if (!module) {
    PyErr_SetString(PyExc_ImportError, "Could not import 'gasp' module");
    return NULL;
  }

  // Create a new empty list
  new_list = PyList_New(0);
  if (!new_list) {
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not create new list for grids");
    return NULL;
  }

  // Replace the module-level attribute
  if (PyModule_AddObject(module, "grids", new_list) < 0) {
    Py_DECREF(new_list);
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'grids' attribute to module");
    return NULL;
  }

    Py_DECREF(module);
    Py_RETURN_NONE;
}





PyObject* populate_grids_list(int L, int ngrids, int* grid_data) {

  PyObject *module, *list;
  
  // Import the module to get the list
  module = PyImport_ImportModule("gasp");
  if (!module) {
    PyErr_SetString(PyExc_ImportError, "Could not import 'gasp' module");
    return NULL;
  }

  list = PyObject_GetAttrString(module, "grids");
  Py_DECREF(module);
  if (!list) {
    PyErr_SetString(PyExc_AttributeError, "Could not get 'grids' attribute from module");
    return NULL;
  }
  
  for (int g = 0; g < ngrids; ++g) {
    npy_intp dims[2] = {L, L};
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_INT32);
    if (!array) {
      Py_DECREF(list);
      PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for grid");
      return NULL;
    }
    
    int* arr_data = (int*)PyArray_DATA((PyArrayObject*)array);
    memcpy(arr_data, grid_data + g * L * L, L * L * sizeof(int));
    
    if (PyList_Append(list, array) < 0) {
      Py_DECREF(array);
      Py_DECREF(list);
      PyErr_SetString(PyExc_RuntimeError, "Could not append array to grids list");
      return NULL;
    }
    
    Py_DECREF(array);  // PyList_Append increments ref count

  } // grids
  
  Py_DECREF(list);
  return NULL;
    
}

int append_grids_list(int L, int ngrids, int* grid_data, int isweep, float* magnetisation, float *lclus_size, char *cv,double dn_thr, double up_thr) {

  // Update module level lists of magnetisation and largest cluster (if computed)
  float *cv_val;
  if (strcmp(cv, "magnetisation") == 0) {
    cv_val = magnetisation;
  } else if (strcmp(cv, "lclus_size") == 0) {
    cv_val = lclus_size;
  } else {
    cv_val = NULL;
  }

  // Get a handle to the gasp module
  PyObject* module = PyImport_AddModule("gasp"); 
  if (!module) {
    PyErr_SetString(PyExc_RuntimeError, "Could not import module 'gasp' when appending to record lists");
    return -1;
  }
  //printf("Imported gasp module\n");
  
  // Convert C array of current grid magnetisations to a Python list
  PyObject* magnetisation_pylist = PyList_New(ngrids);
  if (!magnetisation_pylist) {
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not create Python list for magnetisation");
    return -1;
  }
  for (int i = 0; i < ngrids; ++i) {
    PyObject* val = PyFloat_FromDouble((double)magnetisation[i]);
    if (!val) {
      Py_DECREF(module);
      Py_DECREF(magnetisation_pylist);
      PyErr_SetString(PyExc_RuntimeError, "Could not create Python float object for magnetisation list");
      return -1;
    }
    PyList_SET_ITEM(magnetisation_pylist, i, val);
  }

  // Append to module-level magnetisation list
  if (module) {
    PyObject* magnetisation_list = PyObject_GetAttrString(module, "magnetisation");
    if (magnetisation_list && PyList_Check(magnetisation_list)) {
      PyList_Append(magnetisation_list, magnetisation_pylist);
    }

    Py_XDECREF(magnetisation_list);
  }
  Py_DECREF(magnetisation_pylist);

  if (lclus_size) { // We don't always record largest cluster size

    // Convert C array lclus_size to Python list
    PyObject* largest_cluster_pylist = PyList_New(ngrids);
    if (!largest_cluster_pylist) {
      return -1;
    }
  
    for (int i = 0; i < ngrids; ++i) {
      PyObject* val = PyFloat_FromDouble((double)lclus_size[i]);
      if (!val) {
        Py_DECREF(largest_cluster_pylist);
        return -1;
      }
      PyList_SET_ITEM(largest_cluster_pylist, i, val);
    }

    // Append to module-level largest_cluster list
    if (module) {
      PyObject* largest_cluster_list = PyObject_GetAttrString(module, "largest_cluster");
      if (largest_cluster_list && PyList_Check(largest_cluster_list)) {
        PyList_Append(largest_cluster_list, largest_cluster_pylist);
      }
      Py_XDECREF(largest_cluster_list);
    }
    Py_DECREF(largest_cluster_pylist);

  } // end if recording largest cluster size

  // Always call the hdf5 writer
  int iret = write_ising_grids_hdf5(L, ngrids, grid_data, isweep, magnetisation, lclus_size, cv, dn_thr, up_thr);

  // If we're maintaining a history of grids in RAM then proceed, otherwise we're done
  if (!grid_history) {
    //Py_DECREF(module);
    return iret;
  }

  // Dimensions of NumPy array member of GridSnapObjects
  npy_intp dims[2] = {L, L};

  // Create a Python list to hold GridSnapObject instances corresponding to the current sweep
  PyObject* grid_snap_list = PyList_New(0);
  if (!grid_snap_list) return -1;

  int snapsize = ngrids*L*L;
  for (int i = 0; i < ngrids; ++i) {
    // Copy current grid data into history array. Not using memcpy as there's a cast involved.
    int isite;
    for (isite=0;isite<L*L;++isite){
      grid_history[snapsize*ihist+i*L*L+isite] = (int8_t)grid_data[i*L*L+isite];
    }
    // Create a NumPy array from the data. This just wraps the grid_history_array
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_INT8, (int8_t*)(grid_history + ihist*snapsize + i * L * L));
    if (!array) {
      PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array from grid\n");
      Py_DECREF(grid_snap_list);
      return -1;
    }
    Py_INCREF(Py_None);
    PyArray_SetBaseObject((PyArrayObject*)array, Py_None);
    // Create GridSnapObject instance
    PyObject* args = Py_BuildValue("iiO", isweep, i, array);
    PyObject* snap_obj = PyObject_CallObject((PyObject*)&GridSnapObjectType, args);
    Py_DECREF(args);
    Py_DECREF(array);
    if (!snap_obj) {
      PyErr_SetString(PyExc_RuntimeError, "Could not create GridSnapObject instance for current grid\n");
      Py_DECREF(grid_snap_list);
      return -1;
    }
    // Set additional attributes
    PyObject_SetAttrString(snap_obj, "magnetisation", PyFloat_FromDouble((double)magnetisation[i]));
    if (lclus_size) {
      PyObject_SetAttrString(snap_obj, "lclus_size", PyFloat_FromDouble((double)lclus_size[i]));
    }
    PyObject_SetAttrString(snap_obj, "committor", PyFloat_FromDouble(-1.0)); // Default value
    //PyObject_SetAttrString(snap_obj, "isweep", PyLong_FromLong(isweep));
    //PyObject_SetAttrString(snap_obj, "grid_index", PyLong_FromLong(i));

    // Append if the selected CV lies between up_thr and down_thr
    //if (cv_val[i] > dn_thr && cv_val[i] < up_thr) {
      PyList_Append(grid_snap_list, snap_obj);
    //}
  }


  PyObject* existing_list = PyObject_GetAttrString(module, "grids");
  if (!existing_list || !PyList_Check(existing_list)) {
    Py_XDECREF(existing_list);
    Py_DECREF(grid_snap_list);
    PyErr_SetString(PyExc_RuntimeError, "Attribute 'grids' not found or not a list");
    return -1;
  }
  // Append the new list of GridSnapObject instances only if grid_snap_list is not empty
  if (PyList_Size(grid_snap_list) > 0 ) {
    if (PyList_Append(existing_list, grid_snap_list) < 0) {
      Py_DECREF(existing_list);
      Py_DECREF(grid_snap_list);
      return -1;
    }
  }
  Py_DECREF(existing_list);
  Py_DECREF(grid_snap_list);

  //Py_DECREF(module);

  ihist++; // Increment snapshot history counter
  
  return 0;
  
}


// Corresponds to itask = 0 in the original ising.cu
static PyObject* method_run_nucleation_swarm(PyObject* self, PyObject* args, PyObject* kwargs){

  //unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
  unsigned long rngseed = (long)time(NULL);
  
  /* Positional arguments to extract */
  
  int L = 64;             // Size of LxL grid
  int ngrids = 128;       // Number of grids in the swarm
  int tot_nsweeps = 100;  // Number of sweeps to run for each grid

  double beta = 0.54;     // inverse temperature
  double h = 0.07;        // magnetic field


  /* Keyword arguments to extract */

  int initial_spin = -1;         // Majority spin in parent phase
   
  double up_threshold = -0.90*(double)initial_spin;  // Threshold mag at which which assumed reversed
  double dn_threshold =  0.95*(double)initial_spin;  // Threshold mag at which assumed returned

  int mag_output_int = 100;      // Sweeps between output of magnetisation
  int grid_output_int = 1000;    // Sweeps between output of grids
  int keep_grids = 1;            // Keep history of grids in RAM

  int threadsPerBlock = 32;      // Threads per block
  int gpu_method = 0;            // GPU method to use - see mc_gpu.cu

  char *cv="magnetisation";     // Collective variable to use in determining fate

  /* list of keywords */
  static char* kwlist[] = {"L", "ngrid", "tot_nsweeps", "beta", "h",
    "initial_spin", "cv", "up_threshold", "dn_threshold", "mag_output_int",
    "grid_output_int", "keep_grids", "threadsPerBlock", "gpu_method", NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiidd|isddiipii", kwlist,
                     &L, &ngrids, &tot_nsweeps, &beta, &h, &initial_spin, &cv,
                     &up_threshold, &dn_threshold, &mag_output_int,
                     &grid_output_int, &keep_grids, &threadsPerBlock, &gpu_method)) {
      return NULL;
    }


  // Validate cv argument
  if (strcmp(cv, "magnetisation") != 0 && strcmp(cv, "largest_cluster") != 0) {
    PyErr_SetString(PyExc_ValueError, "cv must be either 'magnetisation' or 'largest_cluster'");
    return NULL;
  }

  if (strcmp(cv, "largest_cluster") == 1) {
    // Warn if magnitude of up_threshold and dn_threshold are less than 2
    if (fabs(up_threshold) < 2.0 || fabs(dn_threshold) < 2.0) {
      PyErr_SetString(PyExc_UserWarning, "For cv='largest_cluster', up_threshold and dn_threshold should have magnitudes >= 2");
    }
  }

  /*=================================
   Delete old output 
  ================================*/
  remove("gridstates.bin");

  /* Reset the module level list of grids to empty */
  reset_grids_list(self, NULL);

  /* Allocate the C array this list of grids wraps */
  if (grid_history != NULL) { free(grid_history); }  
  int nsnaps = tot_nsweeps/grid_output_int + 1;
  if (keep_grids) {
    grid_history = (int8_t *)malloc(nsnaps*ngrids*L*L*sizeof(int8_t));
    if (grid_history == NULL){
      PyErr_SetString(PyExc_MemoryError, "Error allocating RAM to hold grid history!");
      return NULL;   
    }
  }
  ihist = 0;
  
  


/*=================================
   Initialise simulations
  =================================*/ 
  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)
  
  
  
  
  // Initialise as 100% spin down for all grids
  ising_grids = init_grids_uniform(L, ngrids, initial_spin);
  grid_fate = NULL ; // not used

  float *result = (float *)malloc((tot_nsweeps/mag_output_int) * sizeof(float)); // fraction nucleated at each mag_output_int
  if (result == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for result array");
    free(ising_grids);
    if (grid_fate) free(grid_fate);
    return NULL;
  }


/*=================================
    Run simulations - CPU version
  =================================*/ 

  if (run_cpu==true) {


    fprintf(stdout, "Using CPU\n");
    
    // Initialise host RNG
    init_genrand(rngseed);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_cpu(beta, h);

    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 0; calc.cv = cv; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = 1; calc.result = result;

    /*=================================
      Write output header 
      ================================*/
#ifndef PYTHON
    fprintf(stdout, "# isweep    nucleated fraction\n");
#endif
    
    // Perform the MC simulations
    //mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, write_ising_grids);
    mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, append_grids_list);
    
  }

/*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){


    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    // Initialise model grid on GPU
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    // Select gpu_method 
    //printf("Calling select_gpu_method\n");
    gpu_method = select_gpu_method(L, ngrids, threadsPerBlock, idev);
    
    curandState *d_state;                  // Pointer to device RNG states

    // Initialise RNG on GPU
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);


    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.initial_spin = initial_spin; calc.cv = cv; calc.itask = 0; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = 1; calc.result = result;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method;

    /*=================================
      Write output header 
      ================================*/
#ifndef PYTHON
    fprintf(stdout, "# isweep    nucleated fraction\n");
#endif
    create_ising_grids_hdf5(L, ngrids, tot_nsweeps, h, beta);
    //result = mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, write_ising_grids);
    mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, append_grids_list);
    //mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, write_ising_grids_hdf5);

    // Free device arrays
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_neighbour_list) );

 }

  //populate_grids_list(L, ngrids, ising_grids);
  
/*=================================================
    Tidy up memory used in both GPU and CPU paths
  =================================================*/ 
  if (ising_grids) free(ising_grids);

  // Return a NumPy array holding the contents of the C array 'result'
  npy_intp dims[1] = {tot_nsweeps/mag_output_int};
  PyObject* result_array = PyArray_SimpleNew(1, dims, NPY_FLOAT);
  if (!result_array) {
    if (result) free(result);
    return NULL;
  }
  float* result_data = (float*)PyArray_DATA((PyArrayObject*)result_array);
  for (int i = 0; i < dims[0]; ++i) {
    result_data[i] = result[i];
  }
  if (result) free(result);
  return result_array;


}


static PyObject* method_run_committor_calc(PyObject* self, PyObject* args, PyObject* kwargs){

  unsigned long rngseed = (long)time(NULL);

  int L = 64;
  int ngrids = 128;
  int tot_nsweeps = 100;
  double beta = 0.54;
  double h = 0.07;
  int initial_spin = -1;         // Majority spin in parent phase   
  double up_threshold = -0.90*(double)initial_spin;  // Threshold mag at which which assumed reversed
  double dn_threshold =  0.93*(double)initial_spin;  // Threshold mag at which assumed returned
  int mag_output_int = 100;
  int grid_output_int = 1000;
  int keep_grids = 1;
  int threadsPerBlock = 32;
  int gpu_method = 0;
  const char* grid_input = "gridstates.bin";
  PyObject* grid_array_obj = NULL;
  char* cv = "magnetisation";

  static char* kwlist[] = {"L", "ngrid", "tot_nsweeps", "beta", "h",
    "initial_spin", "cv", "up_threshold", "dn_threshold", "mag_output_int",
    "grid_output_int", "keep_grids", "threadsPerBlock", "gpu_method", "grid_input", "grid_array", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiidd|isddiipiisO", kwlist,
       &L, &ngrids, &tot_nsweeps, &beta, &h, &initial_spin, &cv,
       &up_threshold, &dn_threshold, &mag_output_int,
       &grid_output_int, &keep_grids, &threadsPerBlock, &gpu_method, &grid_input, &grid_array_obj)) {
    return NULL;
  }

  // Validate cv argument
  if (strcmp(cv, "magnetisation") != 0 && strcmp(cv, "largest_cluster") != 0) {
    PyErr_SetString(PyExc_ValueError, "cv must be either 'magnetisation' or 'largest_cluster'");
    return NULL;
  }

  int* grid_array_c = NULL;
  int grid_array_count = 1; // Default to 1 if no grid_array provided
  if (grid_array_obj && grid_array_obj != Py_None) {
    if (!PyList_Check(grid_array_obj)) {
      PyErr_SetString(PyExc_TypeError, "grid_array must be a list of 2D NumPy arrays");
      return NULL;
    }
    grid_array_count = (int)PyList_Size(grid_array_obj);
    if (grid_array_count > gpu_nsms) {
      PyErr_SetString(PyExc_ValueError, "grid_array list length exceeds gpu_nsms");
      return NULL;
    }
    if (grid_array_count == 0) {
      PyErr_SetString(PyExc_ValueError, "grid_array list is empty");
      return NULL;
    }
    if (gpu_nsms % grid_array_count != 0) {
      PyErr_SetString(PyExc_ValueError, "grid_array list length must divide gpu_nsms exactly");
      return NULL;
    }
    if (ngrids % gpu_nsms != 0) {
      PyErr_SetString(PyExc_ValueError, "ngrids must be a multiple of gpu_nsms");
      return NULL;
    }
    if (ngrids % grid_array_count != 0) {
      PyErr_SetString(PyExc_ValueError, "ngrids must be a multiple of grid_array list length");
      return NULL;
    }

    // Allocate space for all arrays
    grid_array_c = (int*)malloc(grid_array_count * L * L * sizeof(int));
    if (!grid_array_c) {
      PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for grid_array_c");
      return NULL;
    }
    for (int arr_idx = 0; arr_idx < grid_array_count; ++arr_idx) {
      PyObject* arr_obj = PyList_GetItem(grid_array_obj, arr_idx);
      if (!PyArray_Check(arr_obj)) {
        PyErr_SetString(PyExc_TypeError, "All items in grid_array must be NumPy arrays");
        free(grid_array_c);
        return NULL;
      }
      PyArrayObject* arr = (PyArrayObject*)arr_obj;
      if (PyArray_TYPE(arr) != NPY_INT8 || PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_TypeError, "Each grid_array item must be a 2D NumPy array of type int8");
        free(grid_array_c);
        return NULL;
      }
      int nrows = (int)PyArray_DIM(arr, 0);
      int ncols = (int)PyArray_DIM(arr, 1);
      if (nrows != L || ncols != L) {
        PyErr_SetString(PyExc_ValueError, "Each grid_array item must have shape LxL");
        free(grid_array_c);
        return NULL;
      }
      npy_int8* arr_data = (npy_int8*)PyArray_DATA(arr);
      for (int i = 0; i < L * L; ++i) {
        grid_array_c[arr_idx * L * L + i] = (int)arr_data[i];
      }
    }
  }

/*=================================
   Delete old output 
  ================================*/
  remove("gridstates.bin");

  /* Reset the module level list of grids to empty */
  reset_grids_list(self, NULL);

  /* Allocate the C array this list of grids wraps */
  if (grid_history != NULL) { free(grid_history); }  
  int nsnaps = tot_nsweeps/grid_output_int + 1;
  if (keep_grids){
    grid_history = (int8_t *)malloc(nsnaps*ngrids*L*L*sizeof(int8_t));
    if (grid_history == NULL){
      PyErr_SetString(PyExc_MemoryError, "Error allocating RAM to hold grid history!");
      return NULL;   
    }
  }
  ihist = 0;
  
  


/*=================================
   Initialise simulations
  =================================*/ 
  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)
  
  
  // Initialise as 100% spin down for all grids
  //ising_grids = init_grids_uniform(L, ngrids, initial_spin);
    if (strcmp(grid_input, "gridstates.bin") == 0) {
      ising_grids = init_grids_from_file(L, ngrids); // read from gridinput.bin
    } else if ((strcmp(grid_input, "NumPy") == 0) && grid_array_obj != NULL) {
      ising_grids = init_grids_from_array(L, ngrids, grid_array_count, grid_array_c); // read from the supplied array
    } else {
      PyErr_SetString(PyExc_ValueError, "Invalid grid_input option or no grid_array supplied");
      return NULL;
    }


  grid_fate = init_fates(ngrids); // grid fates

  float *result = (float *)malloc(grid_array_count * sizeof(float)); // result of committor calcs
  if (result == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for result array");
    free(ising_grids);
    if (grid_fate) free(grid_fate);
    if (grid_array_c) free(grid_array_c);
    return NULL;
  }

  double *errbar = (double *)malloc(grid_array_count * sizeof(double)); // error bar on committor estimates
  if (errbar == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for error bar array");
    free(ising_grids);
    free(grid_fate);
    if (grid_array_c) free(grid_array_c);
    free(result);
    return NULL;
  }

/*=================================
    Run simulations - CPU version
  =================================*/ 

  if (run_cpu==true) {


#ifndef PYTHON
    fprintf(stdout,"Using CPU\n");
#endif

    // Initialise host RNG
    init_genrand(rngseed);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_cpu(beta, h);

    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 1; calc.initial_spin = initial_spin; calc.cv = cv; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = grid_array_count; calc.result=result;

    
    // Perform the MC simulations
    //result = mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, write_ising_grids);
    mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, append_grids_list);
    
  }

/*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){


    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    // Initialise model grid on GPU
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    // Select gpu_method 
    //printf("Calling select_gpu_method\n");
    gpu_method = select_gpu_method(L, ngrids, threadsPerBlock, idev);
    
    curandState *d_state;                  // Pointer to device RNG states

    // Initialise RNG on GPU
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);

    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 1; calc.cv = cv; calc.initial_spin = initial_spin; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = grid_array_count; calc.result=result;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method; 


    //result = mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, write_ising_grids);
    mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, append_grids_list);
    
    // Free device arrays
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_neighbour_list) );

 }


/* =================================================
    Calculate error bar on committor estimates
  ================================================== */
  for (int i = 0; i < grid_array_count; ++i) {
    errbar[i] = bootstrap_errbar(ngrids/grid_array_count, ngrids/grid_array_count, grid_fate+(i * ngrids/grid_array_count));
    if (errbar[i] < 0.0) {
      PyErr_SetString(PyExc_RuntimeError, "Error calculating bootstrap error bar");
      free(ising_grids);
      free(grid_fate);
      if (grid_array_c) free(grid_array_c);
      free(result);
      free(errbar);
      return NULL;
    }    
  }
  //populate_grids_list(L, ngrids, ising_grids);
  
/*=================================================
    Tidy up memory used in both GPU and CPU paths
  =================================================*/ 
  free(ising_grids);
  free(grid_fate);
  if (grid_array_c) free(grid_array_c);

  //return PyFloat_FromDouble((double)result);

  // Convert result array to a list of Python tuples
  // Second entry in the tuple will hold the uncertainty on the pB estimate
  PyObject* pylist = PyList_New(grid_array_count);
  if (!pylist) {
    if (result) free(result);
    return NULL;
  }
  for (int i = 0; i < grid_array_count; ++i) {
    PyObject* tuple = PyTuple_New(2);
    if (!tuple) {
      Py_DECREF(pylist);
      if (result) free(result);
      return NULL;
    }
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble((double)result[i]));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(errbar[i]));
    PyList_SET_ITEM(pylist, i, tuple);
  }
  if (result) free(result);
  if (errbar) free(errbar);
  
  return pylist;


}



static PyMethodDef GPUIsingMethods[] = {
  {"run_committor_calc", (PyCFunction)method_run_committor_calc, METH_VARARGS | METH_KEYWORDS, "DocString placeholder for committor calc!"},
  {"run_nucleation_swarm", (PyCFunction)method_run_nucleation_swarm, METH_VARARGS | METH_KEYWORDS, "DocString placeholder!"},
  {NULL, NULL, 0, NULL} /* Sentinal */
};

/* Defines the Python module that our wrapped function will live inside */
static struct PyModuleDef gaspmodule = {
    PyModuleDef_HEAD_INIT,   /* Always this https://docs.python.org/3/c-api/module.html#initializing-c-modules */
    "gasp",                  /* Name of the module (same as name of wrapped function in this case */
    "Python interface to the GASP code for GPU accelerated 2D Ising Model", /* DocString for the module */
    -1,                      /* Memory required (bytes) for each instance, or -1 if multiple instances not supported */
    GPUIsingMethods,         /* Name of the PyMethodDef object to use for the list of member functions */
};

/* The init function for the module */
PyMODINIT_FUNC PyInit_gasp(void) { 

   // Initialize NumPy API
  import_array();
  
  /* Register the custom type we'll use to pass grids back to Python */
  if (PyType_Ready(&GridSnapObjectType) < 0) {
      return NULL;
  }

  /* Assign created module to a variable */
  PyObject* module = PyModule_Create(&gaspmodule);
  if (!module) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create gasp module");
    return NULL;
  }

  /* Add the custom type to the module */
  Py_INCREF(&GridSnapObjectType);
  if (PyModule_AddObject(module, "GridSnapObject", (PyObject *) &GridSnapObjectType) < 0) {
      Py_DECREF(&GridSnapObjectType);
      Py_DECREF(module);
      return NULL;
  }

  /* Initialise GPU device if available on import */
  if (run_gpu==true) {

    // Initialise GPU device(s)
    idev = gpuInitDevice(-1, &gpu_nsms); 
    if (idev==-1){
      printf("Falling back to CPU\n");
      run_cpu=true;
      run_gpu=false;
    }
    
  }

  /* Create a module level object to hold the history of grids */
  PyObject* list = PyList_New(0);
  if (!list) {
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    return NULL;
  }

  // Set the list as a module-level attribute called grids
  if (PyModule_AddObject(module, "grids", list) < 0) {
    Py_DECREF(list);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'grids' attribute to gasp module");
    return NULL;
  }


  // Add gpu_nsms as a module-level integer attribute
  PyObject* py_gpu_nsms = PyLong_FromLong((long)gpu_nsms);
  if (PyModule_AddObject(module, "gpu_nsms", py_gpu_nsms) < 0) {
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'gpu_nsms' attribute to gasp module");
    return NULL;
  }
  
  // Add module-level lists for magnetisation and largest_cluster
  PyObject* magnetisation_list = PyList_New(0);
  if (!magnetisation_list) {
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    return NULL;
  }
  if (PyModule_AddObject(module, "magnetisation", magnetisation_list) < 0) {
    Py_DECREF(magnetisation_list);
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'magnetisation' attribute to gasp module");
    return NULL;
  }

  PyObject* largest_cluster_list = PyList_New(0);
  if (!largest_cluster_list) {
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    return NULL;
  }
  if (PyModule_AddObject(module, "largest_cluster", largest_cluster_list) < 0) {
    Py_DECREF(largest_cluster_list);
    Py_DECREF(magnetisation_list);
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'largest_cluster' attribute to gasp module");
    return NULL;
  }




  return module;
    
}