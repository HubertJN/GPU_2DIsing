import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from modules.dataset import load_hdf5_raw, to_cnn_dataset
from modules.architecture import CNN

# --- Config ---
h5_path = "training_gridstates.hdf5"
checkpoint_path = "models/cnn.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_grids = 36  # Number of lowest-committor grids to plot (user-configurable)
figsize_per_grid = 2  # Size of each grid in inches

# --- Load dataset ---
grids, attrs = load_hdf5_raw(h5_path)
committors = attrs[:, 2]      # committor
magnetizations = attrs[:, 0]  # magnetization
cluster_sizes = attrs[:, 1]   # largest cluster size

# --- Prepare CNN dataset for predictions ---
dataset_cnn = to_cnn_dataset(grids, attrs, device="cpu")  # keep on CPU for batch processing

# --- Load trained CNN model ---
model = CNN(input_size=64, channels=16, num_cnn_layers=4, num_fc_layers=1, dropout=0.1).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Find indices of lowest committors ---
lowest_idx = np.argsort(committors)[:num_grids]
lowest_grids = grids[lowest_idx]
lowest_comms = committors[lowest_idx]
lowest_mags = magnetizations[lowest_idx]
lowest_clusters = cluster_sizes[lowest_idx]

# --- Compute CNN predictions for these grids ---
predictions = []
with torch.no_grad():
    for idx in lowest_idx:
        x, _ = dataset_cnn[idx]
        x = x.unsqueeze(0).to(device)  # add batch dimension
        pred = model(x).squeeze().item()
        predictions.append(pred)
predictions = np.array(predictions)

# --- Compute layout (rows x cols) ---
cols = math.ceil(math.sqrt(num_grids))
rows = math.ceil(num_grids / cols)
figsize = (cols * figsize_per_grid, rows * figsize_per_grid)

# --- Plot grids ---
fig, axs = plt.subplots(rows, cols, figsize=figsize)
axs = axs.flatten()

for i, ax in enumerate(axs):
    if i < num_grids:
        grid = lowest_grids[i]
        comm = lowest_comms[i]
        mag = lowest_mags[i]
        cluster = lowest_clusters[i]
        pred = predictions[i]
        ax.imshow(grid, cmap="gray", interpolation="none")
        ax.set_title(f"C={comm:.3f}\nM={mag:.3f}\nCluster={cluster:.1f}\nPred={pred:.3f}", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"{num_grids} Grids with Lowest Committor (with CNN Predictions)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"lowest_{num_grids}_committor_grids.pdf")
print(f"Saved figure to lowest_{num_grids}_committor_grids.pdf")
plt.close()
