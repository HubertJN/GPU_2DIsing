import h5py
import numpy as np
import os
from tools.utils import load_config

config = load_config("config.yaml")
num_tasks = config.gpu.tasks  # same as your SLURM array size

all_attrs = []

for task_id in range(num_tasks):
    temp_file = os.path.join(config.paths.save_dir, f"attrs_task{task_id}.h5")
    with h5py.File(temp_file, "r") as fo:
        all_attrs.append(fo["attrs"][:])

# Concatenate along the first axis
merged_attrs = np.vstack(all_attrs)

# Write back to original training file
with h5py.File(config.paths.training, "r+") as fo:
    fo["attrs"][:, :] = merged_attrs

print(f"Merged {num_tasks} temporary files into {config.paths.training}")
