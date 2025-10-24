import os
import argparse
import numpy as np
import h5py
import gasp
from tools.utils import load_into_array, load_config

# --- parse optional inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
args = parser.parse_args()

# --- load config ---
config = load_config("config.yaml")

# --- apply overrides if provided ---
beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h

# --- load dataset (path depends on beta/h) ---
training_path = f"data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"
grids, attrs, headers = load_into_array(training_path)

L = headers["L"]
nsweeps = headers["tot_nsweeps"]

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult
ngrids = 4 * gpu_nsms * 32
conc_calc = min(int(np.round(ngrids / config.comm.ngrids)), gpu_nsms)

# --- SLURM setup ---
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", config.gpu.tasks))

task_id = task_id%num_tasks

chunk_size = (len(grids) + num_tasks - 1) // num_tasks
start_idx = task_id * chunk_size
end_idx = min((task_id + 1) * chunk_size, len(grids))

print(f"[Task {task_id}] Processing grids {start_idx}:{end_idx} of {len(grids)}")

# Slice dataset
grids = grids[start_idx:end_idx]
attrs = attrs[start_idx:end_idx]

cluster = attrs[:, 1]

cluster_min = config.analyse.cluster_min
cluster_max = config.analyse.cluster_max
bins = np.arange(cluster_min - 0.5, cluster_max + 1.5, 1) if config.analyse.bin_per_cluster else config.analyse.bins

counts, bin_edges = np.histogram(cluster, bins=bins)
max_idx = np.argmax(counts)
bin_center = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2

up_threshold = config.collective_variable.up_threshold
dn_threshold = bin_center - 0.5

# --- output path (depends on beta/h) ---
outpath = os.path.join(
    config.paths.save_dir,
    f"attrs_task{task_id}_{beta:.3f}_{h:.3f}.h5"
)

total_rows = len(grids)
ncols = attrs.shape[1]

print(f"Running {conc_calc} concurrent calculation on {int(ngrids/conc_calc)} grids each")
exit()

# --- main computation ---
with h5py.File(outpath, "w") as fo:
    dset = fo.create_dataset("attrs", shape=(total_rows, ncols), dtype=attrs.dtype)
    offset = 0

    for local_start in range(0, len(grids), conc_calc):
        local_end = min(local_start + conc_calc, len(grids))
        gridlist = [g.copy() for g in grids[local_start:local_end]]
        attrlist = attrs[local_start:local_end].copy()

        pBfast = np.array(
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
            )
        )

        attrlist[:, 2] = pBfast[:, 0]
        attrlist[:, 3] = pBfast[:, 1]
        dset[offset:offset + (local_end - local_start), :] = attrlist
        offset += (local_end - local_start)

print(f"[Task {task_id}] Finished writing {total_rows} grids to {outpath}")
