import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import h5py
import re

from modules.architecture import CNN, GNN, data_to_dataloader
from modules.dataset import load_hdf5_raw, to_cnn_dataset, to_gnn_dataset, uniform_filter

# --- Config ---
model_type = "cnn"
h5path = "../Jupyter/gridstates.hdf5"
save_path = f"models/{model_type}.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load full dataset ---
print("Loading dataset...")
with h5py.File(h5path, 'r') as f:
    total_saved = int(f['total_saved_grids'][()])
    L = int(f['L'][()])

    print(f"Header: total_saved_grids={total_saved}, L={L}")

    grids = np.empty((total_saved, L, L), dtype=np.int8)  # smallest int type
    attrs = np.empty((total_saved, 4), dtype=np.float64)  # 4 float attrs per grid

    idx = 0
    for sweep_name in sorted((k for k in f.keys() if k.startswith('sweep_')), key=lambda x: int(re.search(r'\d+', x).group())):
        print(sweep_name)
        grp = f[sweep_name]
        for dname in sorted(grp.keys()):
            if idx >= total_saved:
                break

            data = grp[dname][()]

            # Dataset is a 1D packed byte array: unpack bits to +1/-1 grid.
            b = np.asarray(data, dtype=np.uint8).ravel()
            nbits = L * L
            flat = np.empty(nbits, dtype=np.int8)
            for i in range(nbits):
                byte_idx = i // 8
                bit_idx = i % 8
                bit = (b[byte_idx] >> bit_idx) & 1
                flat[i] = 1 if bit == 1 else -1

            grids[idx] = flat.reshape((L, L))

            idx += 1

    grids = grids[:idx]

    print(f"Loaded {idx} grids. grids.shape={grids.shape}, attrs.shape={attrs.shape}")

# Choose dataset type based on model
if model_type == "cnn":
    dataset = to_cnn_dataset(grids, attrs, device)
else:
    dataset = to_gnn_dataset(grids, attrs, device)

print(f"Full dataset size: {len(dataset)}")

# --- Initialize model and load weights ---
if model_type == "cnn":
    model = CNN(channels=16, num_cnn_layers=4, num_fc_layers=1, dropout=0.3).to(device)
else:
    model = GNN().to(device)

model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
print(f"Loaded pretrained {model_type.upper()} model from {save_path}")

# --- Run predictions ---
with torch.no_grad():
    #grids = torch.tensor(grids, dtype=torch.float32)

    predictions = np.zeros(len(grids))
    for i, image in enumerate(grids):
        image = image.unsqueeze(0).unsqueeze(0)
        pred = model(image.to(device)).item()
        predictions[i] = pred

num_bins = 20
bins = np.linspace(0, 1, num_bins + 1)  # same bins as sampling

# Compute num_to_sample globally as before (smallest non-empty bin)
bin_counts = [np.sum((predictions >= bins[i]) & (predictions < bins[i+1])) for i in range(num_bins)]
num_to_sample = np.min([c for c in bin_counts if c > 0])

subset_indices = []

# Sample num_to_sample points from each bin
for i in range(num_bins):
	bin_idx = np.where((predictions >= bins[i]) & (predictions < bins[i+1]))[0]
	if len(bin_idx) == 0:
		continue
	selected = np.random.choice(bin_idx, size=num_to_sample, replace=False)
	subset_indices.extend(selected)

subset_indices = np.array(subset_indices)
np.save("grid_index", subset_indices)

uniform_samples = predictions[subset_indices]

# --- Plot histogram using the same bins ---
plt.figure(figsize=(8,6))
plt.hist(uniform_samples, bins=bins, color='skyblue', edgecolor='black', rwidth=0.9)
plt.xlabel("predictions")
plt.ylabel("Frequency")
plt.title("Histogram of Uniformly Sampled Predictions")
plt.savfig("histogram.pdf")