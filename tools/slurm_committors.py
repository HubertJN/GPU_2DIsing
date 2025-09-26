import os
import numpy as np
import h5py
import gasp
from tools.utils import load_into_array, load_config

# --- Load config and dataset ---
config = load_config("config.yaml")
grids, attrs, headers = load_into_array(config.paths.training)

L = headers['L']
nsweeps = headers['tot_nsweeps']
beta = headers['beta']
h = headers['h']

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult
ngrids = 4 * gpu_nsms * 32
conc_calc = min(int(np.round(ngrids / config.comm.ngrids)), gpu_nsms)

# --- SLURM array setup ---
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", config.gpu.tasks))
chunk_size = (len(grids) + num_tasks - 1) // num_tasks
start_idx = task_id * chunk_size
end_idx = min((task_id + 1) * chunk_size, len(grids))

print(f"[Task {task_id}] Processing grids {start_idx}:{end_idx} of {len(grids)}")

# Slice dataset for this job
grids = grids[start_idx:end_idx]
attrs = attrs[start_idx:end_idx]

dn_threshold = config.collective_variable.dn_threshold
up_threshold = config.collective_variable.up_threshold

# --- Preallocate HDF5 dataset for this task ---
outpath = os.path.join(config.paths.save_dir, f"attrs_task{task_id}.h5")
total_rows = len(grids)
ncols = attrs.shape[1]

with h5py.File(outpath, "w") as fo:
    dset = fo.create_dataset("attrs", shape=(total_rows, ncols), dtype=attrs.dtype)

    # Process the dataset in batches of size conc_calc
    offset = 0
    for local_start in range(0, len(grids), conc_calc):
        local_end = min(local_start + conc_calc, len(grids))

        gridlist = [g.copy() for g in grids[local_start:local_end]]
        attrlist = attrs[local_start:local_end].copy()

        if attrlist[:, 1].min() < 20:
            mag_output_int = 1
        elif attrlist[:, 1].min() < 40:
            mag_output_int = 10
        else:
            mag_output_int = config.output.mag_output_int

        pBfast = np.array(
            gasp.run_committor_calc(
                L, ngrids, config.comm.nsweeps, beta, h,
                grid_output_int=50000,
                mag_output_int=mag_output_int,
                grid_input="NumPy",
                grid_array=gridlist,
                cv=config.collective_variable.type,
                dn_threshold=dn_threshold,
                up_threshold=up_threshold,
                keep_grids=False,
                nsms=gpu_nsms
            )
        )

        attrlist[:, 2] = pBfast[:, 0]
        attrlist[:, 3] = pBfast[:, 1]

        # Write batch directly into preallocated dataset
        batch_size = local_end - local_start
        dset[offset:offset + batch_size, :] = attrlist
        offset += batch_size

print(f"[Task {task_id}] Finished writing {total_rows} grids to {outpath}")
