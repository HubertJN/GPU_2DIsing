import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tools.utils import load_config, load_into_array, index
import h5py

# --- Load config and grids ---
config = load_config("config.yaml")
h5path = config.paths.gridstates
plot_dir = config.paths.plot_dir

# Load grids
grids, attrs, _ = load_into_array(h5path, load_grids=True)

# Parameters
with h5py.File(h5path, "r") as f:
    ngrids = int(f['ngrids'][()])
    sweeps = int(f['tot_nsweeps'][()]/100)
print(ngrids, sweeps)

grid_index = 0
step = 1  # subsample to reduce frames
selected_sweeps = range(0, sweeps, step)

# Get magnetizations for this grid across selected_sweeps
magnetizations = np.array([attrs[grid_index + sweep * ngrids, 0] 
                           for sweep in selected_sweeps])

# Find indices where magnetization is within [-0.95, 0.95]
valid_mask = (magnetizations > -0.95) & (magnetizations < 0.95)

# Find contiguous segment where it’s valid
if np.any(valid_mask):
    first_valid = np.argmax(valid_mask)  # first True
    # last True after first_valid
    last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
    valid_sweeps = [s for i, s in enumerate(selected_sweeps) if first_valid <= i <= last_valid]
else:
    valid_sweeps = []

# Extract the filtered grid evolution
grid_evolution = np.array([grids[grid_index + sweep * ngrids] for sweep in valid_sweeps])

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
