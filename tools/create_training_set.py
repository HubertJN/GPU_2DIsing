import numpy as np
import h5py
import matplotlib.pyplot as plt
from tools.utils import load_into_array, save_training_grids, load_config
import gasp

# --- Load config ---
config = load_config("config.yaml")

# Load only headers and attributes first
_, attrs, headers = load_into_array("data/gridstates.hdf5", load_grids=False)

target_val = 60
tolerance = abs(0.33*target_val)
sigma = tolerance
size_mult = 0.2
skew = 0

val_min = 1
val_max = 400
num_bins = val_max - val_min + 1#128
max_samples = 5000

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult

values = attrs[:, 1]
mask = (values >= val_min) & (values <= val_max)
candidates, cand_val = np.where(mask)[0], values[mask]
max_samples = min(max_samples, len(attrs))

if len(candidates) == 0:
    sample_idx = np.array([], dtype=int)
else:
    bins = np.linspace(val_min, val_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_counts, _ = np.histogram(cand_val, bins=bins)
    nonempty = bin_counts > 0
    z = (bin_centers - target_val) / sigma
    desired_pdf = np.exp(-0.5 * z**2) * (1 + np.tanh(skew * z)) * size_mult
    #desired_pdf = np.clip(desired_pdf, 1e-12, None)
    #desired_pdf *= nonempty.astype(float)
    desired_pdf += 1/num_bins
    desired_mass = desired_pdf / desired_pdf.sum()
    total_candidates = len(candidates)
    gpu_multiple, cap = gpu_nsms, max_samples
    target_samples = min(total_candidates - (total_candidates % gpu_multiple),
                        cap - (cap % gpu_multiple),
                        total_candidates)
    desired_counts_float = desired_mass * target_samples
    desired_counts = np.floor(desired_counts_float).astype(int)
    remaining = target_samples - desired_counts.sum()
    if remaining > 0:
        frac = desired_counts_float - desired_counts
        order = np.argsort(-frac)
        for i in order:
            if remaining == 0: break
            if desired_counts[i] < bin_counts[i]:
                desired_counts[i] += 1
                remaining -= 1
    chosen = []
    for i in range(num_bins):
        k = desired_counts[i]
        if k <= 0: continue
        if i < num_bins - 1:
            bin_mask = (cand_val >= bins[i]) & (cand_val < bins[i+1])
        else:
            bin_mask = (cand_val >= bins[i]) & (cand_val <= bins[i+1])
        idxs_in_bin = candidates[bin_mask]
        k = min(k, len(idxs_in_bin))
        if len(idxs_in_bin) == 0: continue
        sel = np.random.choice(idxs_in_bin, size=k, replace=False)
        chosen.append(sel)
    sample_idx = np.hstack(chosen) if chosen else np.array([], dtype=int)

    # Ensure final selection is multiple of SMs
    n = len(sample_idx)
    n_trim = n - (n % gpu_nsms)
    if n_trim > 0:
        selected_idx = np.linspace(0, n-1, n_trim, dtype=int)
        sample_idx = sample_idx[selected_idx]

if sample_idx.size > 0:
    # Sort indices before loading
    sample_idx = np.sort(sample_idx)

    # Load only the selected grids and attrs
    sample_grids, _, _ = load_into_array("data/gridstates.hdf5", load_grids=True, indices=sample_idx)
    sample_attrs = attrs[sample_idx]

    # Reorder everything consistently by the second column of attrs
    order = np.argsort(sample_attrs[:, 1])
    sample_idx   = sample_idx[order]
    sample_attrs = sample_attrs[order]
    sample_grids = sample_grids[order]

else:
    sample_grids = []
    sample_attrs = np.empty((0, attrs.shape[1]))

print("Selected", len(sample_grids),
      "Percentage of total", int(len(sample_grids) / len(attrs) * 100), "%")

plt.figure(figsize=(6,4))
plt.hist(values[sample_idx], bins=bins,
         edgecolor='black', rwidth=0.9, alpha=0.6)
expected_counts = (desired_mass / desired_mass.sum()) * len(sample_idx)
plt.plot(bin_centers, expected_counts, 'r--', linewidth=2)
plt.xlabel("Values")
plt.ylabel("Count")
plt.savefig("figures/training_samples.pdf")
plt.close()

save_training_grids("data/gridstates_training.hdf5", sample_grids, sample_attrs, headers)
