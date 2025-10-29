import numpy as np
import h5py
import matplotlib.pyplot as plt
from tools.utils import load_into_array, load_config

# --- Load config and data ---
config = load_config("config.yaml")
grids, attrs, headers = load_into_array(config.paths.training)

print(f"Loaded {config.paths.training}")

# --- Extract committor values ---
committor = attrs[:, 2]
committor_error = attrs[:,3]
print(np.median(committor_error))
cluster = attrs[:,1]
idx = ~np.isnan(committor)
cluster = cluster[idx]
committor = committor[idx]
committor_error = committor_error[idx]

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
plt.errorbar(cluster, committor, yerr=committor_error, ms=2, capsize=2, alpha=0.5, fmt='o', ecolor='gray')
plt.ylabel("Committor Probability (pB)")
plt.xlabel("Cluster")
plt.title("Cluster Size against Committor")
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/cluster_committor.pdf")
print(f"Saved committor histogram to {config.paths.plot_dir}/cluster_committor.pdf")
plt.close()
