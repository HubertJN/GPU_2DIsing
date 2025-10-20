# G A S P

GPU accelerated simulations of the 2D Ising model for nucleation studies. 

GASP exploits the massive parallelism of a GPU to run thousands (or tens of thousands) of realisations of the Ising model concurrently. This is particularly useful for studying the rare event statistics of magnetisation reversal, or of nucleation in the lattice gas when the Ising model is interpretted in that context.

The acryonym GASP is derived from the initial application of the code in studying lattice models of solute precipitation. Specifically it stands for "GPU Accelerated Solute Precipitation", or in longer form "GPU Accelerated Solute Precipitation via Python Monte Carlo for Generative AI Simulations and Parameterisation of Fast Approximate Committor Evaluation". The longer form conveniently abbreviates to GASPyMCGASPFACE.

These days we prefer "GPU Accelerated Spin Physics". 

- [Command line version](#command-line-version)
    * [Dependencies](#dependencies)
    * [Installation](#installation)
    * [Usage](#usage)
    * [`gpu_method`](#gpu_method)
    * [Workflow](#workflow)
- [Python extension (PyGASP)](#python-extension-pygasp)
    * [Dependencies and installation](#dependencies-and-installation)
    * [Usage](#usage-1)
- [Acknowledgements](#acknowledgements)

## Command line version

### Dependencies

The command line version of GASP needs only a relatively recent C compiler and CUDA version. If you plan to use
the `vis_gridstates.py` script to visualise trajectories then Python 3 is required. 

The current version of GASP requires a working HDF5 installation to compile, but does not make use of HDF5 
functionality.

### Installation

1. Clone the repository.

    `git clone https://github.com/dquigley533/GASP.git`

2. Change into the `GASP` subdirectory.

    `cd GASP`

3. Edit the `Makefile` to suit your needs. As a minimum you should updated `NVFLAGS` to match the compute capability of your GPU. If you don't 
know the compute capability of your GPU then consult https://en.wikipedia.org/wiki/CUDA. Settings for compiling/linking against 
HDF5 libraries should be detected automatically via `pkg-config`. If not then consult a responsible adult for help.

4. Build the command line version.

    `make`

### Usage:

This assumes the `Makefile` has built the executable to a location in your `PATH` environment variable.

`gasp nsweeps nreplicas mag_output_int grid_output_int threadsPerBlock gpu_device gpu_method beta h itask`

`nsweeps`         : maximum number of MC sweeps (one sweep is L^2 attempted flips)  
`nreplicas`       : number of independent replicas to run in parallel on the GPU  
`mag_output_int`  : sweeps between checking magnetisation against nucleation/committor thresholds  
`grid_output_int` : sweeps between binary dumps of simulation grids to gridstates.bin  
`threadsPerBlock` : number of threads/replicas per GPU thread block, 32 is standard  
`gpu_device`      : index of GPU device to use, set to zero for single GPU system  
`gpu_method`      : 0, 1 or 2 as described below  
`beta`            : inverse temperature for simulation  
`h`               : magnetic field for simulation  
`itask`           : 0 for reversal from spin down configuration, 1 for committor calculation  

### gpu_method

0 : Uses GPU global memory only. This is slow but allows for much larger L.  
1 : Calculations on each grid are performed on a copy in device shared memory, stored as 1 bit per spin.  
2 : Each warp executes on an LxL array of 32-bit numbers, with each thread in a warp of 32 manipulating its own bit.  

Method 2 is typically fastest but cannot accomodate L much greater than 100. Limitations on shared memory also restrict occupancy (active warps per SM) but this doesn't seem to be the limiting factor to throughput. In most cases running with 3 or 4 warps per SM is optimal but this should be investigated with experimentation.

### Workflow

Run with `itask=0` to generate a swarm of nucleating trajectories. For example, using the K20c (13 SMs) in brigitte.csc.warwick.ac.uk;  

`gasp 50000 1248 100 100 32 0 2 0.54 0.07 0`

which will print the fraction of nucleated trajectores (nucleation threshold is up_threshold in ising.cu) over time, measured in sweeps. This can be fitted to 1-exp(-k(t-t0)) to extract the nucleation rate k and the equilibration time t0.

The swarm can be visualised with;

`python vis_gridstates.py`

using 'p' to pause/play, and arrow keys to move between snapshots (left/right) and replicas (up/down). Pressing 'w' will save the current configuration to `gridinput.bin`. The committor pB for this configuration can then be calculated using;

`gasp 50000 1248 100 100 32 0 2 0.54 0.07 1`

which should be repeated several times to calculate a standard error on pB. Ultimately one will want to automate selection of interesting configurations rather than using the visualisation script. The committor calculation uses dn_threshold and up_threshold (defined in ising.cu) to define the boundaries of the A (initial) and B (reversed) macrostates.

## Python extension (PyGASP)

## Dependencies and installation

Statistically speaking, your Python environment is likely to be a nightmare of highly specific package versions which have been overconstrained by developers of your five most favourite tools. Perhaps you're inside an Anaconda environment and two packages you can't live without aren't available on any Anaconda channel you're licensed to use, but you managed some hacky solution involving environment variables you don't really understand. You're terrified to install anything else, or even breathe near your computer for fear of having to spend the next
two weeks getting back to a working configuration. You definately wrote down all the steps you took last time. Didn't you? Even if you did, the world has moved on and the highly specific version of X needed by package Y is now different to the one needed by package Z and you're out of luck. You're going to be
delayed a while in hitting that next milestone in the "AI Engineering" job you blagged your way into off the back of having once ran the PyTorch MNIST image recognition tutorial in a Jupyter notebook.

Don't be that guy. Use a virtual environment.

1. After cloning the repository as above and entering the GASP director, create a virtual environment.

    `python -m venv .venv`

2. Activate the virtual environment.

    `source .venv/bin/activate`

3. Install PyGASP from source.

    `pip install .`

This should also install all necessary dependencies (except those already needed for the command line version) into your virtual environment. If you plan to use the Jupyter notebook examples then also install `matplotlib` and `jupyter`. Don't forget to `deactivate` the virtual environment when you're ready to return to your usual car crash of conflicting packages held together with hope and optimism. 

## Usage

See the example `.ipynb` notebooks in the `Jupyter` directory.


## Acknowledgements

GASP is heavily inspired by the work of Weigel _et al_ "[Population annealing: Massively parallel simulations in statistical physics](https://iopscience.iop.org/article/10.1088/1742-6596/921/1/012017/pdf)", _J. Phys.: Conf. Ser._ **921** 012017.

Its development was partly funded by EPSRC grant EP/R018820/1. 
