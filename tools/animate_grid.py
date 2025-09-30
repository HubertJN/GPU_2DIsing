import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tools.utils import load_config, load_into_array
import h5py

# --- Load config ---
config = load_config("config.yaml")
h5path = config.paths.gridstates
plot_dir = config.paths.plot_dir

# --- Load attributes only ---
_, attrs, _ = load_into_array(h5path, load_grids=False)

# --- Parameters ---
with h5py.File(h5path, "r") as f:
    ngrids = int(f['ngrids'][()])
    total_sweeps = int(f['tot_nsweeps'][()])

# Choose which grid to animate
grid_index = 0
step = 10  # subsample for animation

# Compute all indices corresponding to this grid across sweeps
all_sweep_indices = np.arange(grid_index, ngrids * total_sweeps, ngrids)
selected_sweeps = all_sweep_indices[::step]  # subsample

# --- Filter by magnetization ---
# Compute magnetization directly from attrs if available
magnetizations = np.array([np.sum(attrs[i, 0]) / attrs[i, 0].size for i in selected_sweeps])  # or compute from grids after loading

# Optional: select contiguous sweeps where magnetization is within [-0.95, 0.95]
valid_mask = (magnetizations > -0.95) & (magnetizations < 0.95)
if np.any(valid_mask):
    first_valid = np.argmax(valid_mask)
    last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
    filtered_indices = selected_sweeps[first_valid:last_valid+1]
else:
    filtered_indices = []

# --- Load only the grids we need ---
if len(filtered_indices) > 0:
    grid_evolution, _, _ = load_into_array(h5path, load_grids=True, indices=filtered_indices)
else:
    grid_evolution = np.empty((0, attrs.shape[1], attrs.shape[1]))  # empty fallback

# Create "bounce" frames (forward then backward)
frames = np.concatenate([np.arange(len(grid_evolution)),
                         np.arange(len(grid_evolution)-2, 0, -1)])

# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(grid_evolution[0], cmap='gray', vmin=-1, vmax=1)
plt.axis('off')

# Update function
def update(frame_idx):
    im.set_data(grid_evolution[frame_idx])
    return [im]

# Animation
anim = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)

# Save GIF
anim.save(f"{plot_dir}/grid_evolution_bounce.gif", writer=PillowWriter(fps=5))
plt.close()
