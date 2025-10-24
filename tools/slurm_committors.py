import os
import argparse
import gasp
from tools.utils import load_into_array, load_config

# --- parse optional inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
parser.add_argument("--dn", type=float, help="Override dn_threshold value")
args = parser.parse_args()

# --- load config ---
config = load_config("config.yaml")

# --- apply overrides ---
beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h
dn_threshold = args.dn if args.dn is not None else config.collective_variable.dn_threshold
up_threshold = config.collective_variable.up_threshold

# --- load dataset ---
training_path = f"data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"
grids, _, headers = load_into_array(training_path)

L = headers["L"]
gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult
ngrids = 4 * gpu_nsms * 32
conc_calc = min(int(round(ngrids / config.comm.ngrids)), gpu_nsms)

# --- SLURM setup ---
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", config.gpu.tasks))
chunk_size = (len(grids) + num_tasks - 1) // num_tasks
start_idx = task_id * chunk_size
end_idx = min((task_id + 1) * chunk_size, len(grids))
grids = grids[start_idx:end_idx]

outpath = os.path.join(config.paths.save_dir, f"attrs_task{task_id}_{beta:.3f}_{h:.3f}.h5")

print(f"[Task {task_id}] Processing grids {start_idx}:{end_idx} of {len(grids)}")
print(f"Saving output to {outpath}")

# --- run committor calculation ---
for local_start in range(0, len(grids), conc_calc):
    local_end = min(local_start + conc_calc, len(grids))
    gridlist = [g.copy() for g in grids[local_start:local_end]]

    gasp.run_committor_calc(
        L, ngrids, config.comm.nsweeps, beta, h,
        grid_output_int=50000,
        mag_output_int=1,
        grid_input="NumPy",
        grid_array=gridlist,
        cv=config.collective_variable.type,
        dn_threshold=dn_threshold,
        up_threshold=up_threshold,
        keep_grids=False,
        nsms=gpu_nsms,
        gpu_method=2,
        outname=outpath
    )

print(f"[Task {task_id}] Finished writing to {outpath}")
