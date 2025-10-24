import argparse
import h5py
import numpy as np
import os
from tools.utils import load_config

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

num_tasks = config.gpu.tasks
all_attrs = []

# --- load per-task files ---
for task_id in range(num_tasks):
    temp_file = os.path.join(
        config.paths.save_dir,
        f"attrs_task{task_id}_{beta:.3f}_{h:.3f}.h5"
    )

    if not os.path.exists(temp_file):
        print(f"Warning: missing file {temp_file}")
        continue

    with h5py.File(temp_file, "r") as fo:
        all_attrs.append(fo["attrs"][:])

# --- concatenate and merge ---
if not all_attrs:
    raise RuntimeError("No attribute files found. Check paths and beta/h values.")

merged_attrs = np.vstack(all_attrs)
print(f"Total attributes to save: {len(merged_attrs)}")

# --- update training file (matches beta/h naming convention) ---
training_path = f"data/gridstates_training_{beta:.2f}_{h:.3f}.hdf5"

with h5py.File(training_path, "r+") as fo:
    if "attrs" not in fo:
        raise KeyError(f"'attrs' dataset not found in {training_path}")
    fo["attrs"][:, :] = merged_attrs

print(f"Merged {len(all_attrs)} temporary files into {training_path}")
