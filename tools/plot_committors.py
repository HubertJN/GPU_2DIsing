import numpy as np
import h5py
import matplotlib.pyplot as plt
from tools.utils import load_into_array, load_config

# --- Load config and data ---
config = load_config("config.yaml")
grids, attrs, headers = load_into_array(config.paths.training)

# --- Extract committor values ---
committor = attrs[:, 2]
cluster = attrs[:,1]
cluster = cluster[~np.isnan(committor)]
committor = committor[~np.isnan(committor)]
# --- Plot histogram ---
plt.figure(figsize=(6, 4))
plt.hist(committor, bins=50, edgecolor="black", alpha=0.7)
plt.xlabel("Committor Probability (pB)")
plt.ylabel("Count")
plt.title("Histogram of Committor Values")
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/committor_histogram.pdf")
print(f"Saved committor histogram to {config.paths.plot_dir}/committor_histogram.pdf")
plt.close()

# --- Plot histogram ---
plt.figure(figsize=(6, 4))
plt.scatter(attrs[:, 1], committor)
plt.ylabel("Committor Probability (pB)")
plt.xlabel("Cluster")
plt.title("Cluster Size against Committor")
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/cluster_committor.pdf")
print(f"Saved committor histogram to {config.paths.plot_dir}/cluster_committor.pdf")
plt.close()
