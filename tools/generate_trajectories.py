import gasp
from tools.utils import load_config

config = load_config("config.yaml")

L = config.simulation.L
ngrids = config.simulation.ngrids_factor * gasp.gpu_nsms * config.simulation.threads_per_block
nsweeps = config.simulation.nsweeps

beta = config.parameters.beta
h = config.parameters.h

grid_output_int = config.output.grid_output_int
mag_output_int = config.output.mag_output_int

frac = gasp.run_nucleation_swarm(
    L,
    ngrids,
    nsweeps,
    beta,
    h,
    grid_output_int=grid_output_int,
    mag_output_int=mag_output_int,
    cv=config.collective_variable.type,
    up_threshold=config.collective_variable.up_threshold,
    dn_threshold=config.collective_variable.up_threshold
)