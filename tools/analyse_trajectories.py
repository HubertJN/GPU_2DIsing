import numpy as np
import h5py
import re
import matplotlib.pyplot as plt
from tools.utils import load_config, load_into_array

# --- Load config ---
config = load_config("config.yaml")
h5path = config.paths.gridstates
plot_dir = config.paths.plot_dir

print("Opening:", h5path)

grids, attrs, _ = load_into_array(h5path)

# --- Prune invalid entries ---
mask = (attrs[:, 0] != -1) & (attrs[:, 1] != 0)
grids = grids[mask]
attrs = attrs[mask]

# --- Attributes array ---
print(f"Processed {grids.shape[0]} grids. grids.shape={grids.shape}, attrs.shape={attrs.shape}")

magnetizations = attrs[:, 0]
cluster = attrs[:, 1]

# --- Magnetization histogram ---
plt.figure(figsize=(12, 6))
plt.hist(magnetizations, bins=config.analyse.bins, color='skyblue', edgecolor='black')
plt.xlabel('Magnetization')
plt.ylabel('Frequency')
plt.title('Histogram of Magnetization Values')
plt.xlim(-1, 1)
xmin, xmax = plt.xlim()
plt.xticks(np.arange(round(xmin, 1), round(xmax + 0.1, 1), 0.1))
plt.tight_layout()
plt.savefig(f"{plot_dir}/magnetization_hist.pdf")
plt.close()

# --- Cluster histogram with integer-centered bins ---
cluster_min = config.analyse.cluster_min
cluster_max = config.analyse.cluster_max

if config.analyse.bin_per_cluster:
    # Each cluster size gets its own bin, centered on the integer value
    bin_edges = np.arange(cluster_min - 0.5, cluster_max + 1.5, 1)
else:
    bin_edges = config.analyse.bins  # fixed number of bins

plt.figure(figsize=(8, 6))
plt.hist(cluster, bins=bin_edges, color='skyblue', edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Values')
plt.xlim(cluster_min - 0.5, cluster_max + 0.5)

plt.tight_layout()
plt.savefig(f"{plot_dir}/cluster_hist.pdf")
plt.close()
