import argparse
import gasp
from tools.utils import load_config

# --- parse optional inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
args = parser.parse_args()

# --- load config ---
config = load_config("config.yaml")

L = config.simulation.L
ngrids = config.simulation.ngrids_factor * gasp.gpu_nsms * config.simulation.threads_per_block
nsweeps = config.simulation.nsweeps

# --- apply overrides if provided ---
beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h

# --- update output name if overrides used ---
outname = config.paths.gridstates
if args.beta is not None or args.h is not None:
    outname = f"data/gridstates_{beta:.3f}_{h:.3f}.hdf5"

grid_output_int = config.output.grid_output_int
mag_output_int = config.output.mag_output_int

# --- run simulation ---
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
    dn_threshold=config.collective_variable.dn_threshold,
    keep_grids=False,
    gpu_method=2,
    outname=outname
)
