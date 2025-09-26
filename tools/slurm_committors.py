import os
import numpy as np
import h5py
import gasp
from tools.utils import load_into_array, load_config

config = load_config("config.yaml")

grids, attrs, headers = load_into_array(config.paths.training)

L = headers['L']
nsweeps = headers['tot_nsweeps']
beta = headers['beta']
h = headers['h']

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult
ngrids = 4 * gpu_nsms * 32
conc_calc = min(int(np.round(ngrids/config.comm.ngrids)), gpu_nsms)

# --- SLURM array setup ---
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 8))  # default 8
chunk_size = (len(grids) + num_tasks - 1) // num_tasks
start_idx = task_id * chunk_size
end_idx = min((task_id + 1) * chunk_size, len(grids))

print(f"[Task {task_id}] Processing grids {start_idx}:{end_idx} of {len(grids)}")

# Slice dataset for this job
grids = grids[start_idx:end_idx]
attrs = attrs[start_idx:end_idx]

dn_threshold = config.collective_variable.dn_threshold
up_threshold = config.collective_variable.up_threshold

for local_start in range(0, len(grids), conc_calc):
    local_end = min(local_start + conc_calc, len(grids))

    gridlist = [g.copy() for g in grids[local_start:local_end]]
    attrlist = attrs[local_start:local_end].copy()

    if attrlist[:, 1].min() < 20:
        mag_output_int = 1
    elif attrlist[:, 1].min() < 40:
        mag_output_int = 10
    else:
        mag_output_int = 100

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

    # --- Save into a temporary file per task ---
    outpath = os.path.join(config.paths.save_dir, f"attrs_task{task_id}.h5")
    with h5py.File(outpath, "a") as fo:
        if "attrs" not in fo:
            fo.create_dataset("attrs", data=attrlist, maxshape=(None, attrs.shape[1]))
        else:
            fo["attrs"].resize((fo["attrs"].shape[0] + attrlist.shape[0]), axis=0)
            fo["attrs"][-attrlist.shape[0]:, :] = attrlist
