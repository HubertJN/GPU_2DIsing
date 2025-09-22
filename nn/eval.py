import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from modules.architecture import CNN, GNN, data_to_dataloader
from modules.dataset import load_hdf5_raw, to_cnn_dataset, to_gnn_dataset, uniform_filter

# --- Command-line arguments ---
parser = argparse.ArgumentParser(description="Run analysis with pretrained CNN or GNN")
parser.add_argument("--model", type=str, choices=["cnn", "gnn"], required=True,
                    help="Choose model type: cnn or gnn (required)")
args = parser.parse_args()

# --- Config ---
model_type = args.model.lower()
h5path = "training_gridstates.hdf5"
save_path = f"models/{model_type}.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
test_size = 0.2

# --- Load full dataset ---
print("Loading dataset...")
grids, attrs = load_hdf5_raw(h5path)

#print("Filtering for uniform committor distribution...")
#subset_indices = uniform_filter(grids, attrs[:,2], num_bins=20)
#grids = grids[subset_indices]
#attrs = attrs[subset_indices]

# Choose dataset type based on model
if model_type == "cnn":
    dataset = to_cnn_dataset(grids, attrs, device)
else:
    dataset = to_gnn_dataset(grids, attrs, device)

print(f"Full dataset size: {len(dataset)}")

# --- Split dataset ---
indices = list(range(len(dataset)))
_, valid_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)
valid_ds = dataset

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
    x_valid = torch.tensor(grids[valid_idx], dtype=torch.float32)
    y_valid = torch.tensor(attrs[valid_idx][:, 2], dtype=torch.float32)

    predictions = np.zeros(len(y_valid))
    correct = 0
    for i, image in enumerate(x_valid):
        image = image.unsqueeze(0).unsqueeze(0)
        pred = model(image.to(device)).item()
        predictions[i] = pred
        if abs(pred - y_valid[i].item()) < 0.05:
            correct += 1

    accuracy = 100 * correct / len(x_valid)

# --- Compute RMSE ---
rmse = np.sqrt(np.mean((y_valid.numpy() - predictions) ** 2))
print("============================================")
print(f"Accuracy for validation set : {accuracy:.2f} %")
print("Number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("RMSE: ", rmse)
print("============================================")

# --- Plot results ---
plt.figure(figsize=(5, 5))
plt.scatter(y_valid, predictions, s=0.5)
plt.plot([0, 1], [0, 1], c="red")
plt.xlim(-0.1, 1.1)
plt.xlabel("Target")
plt.ylabel("Prediction")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', fontsize=10,
bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.savefig("target_prediction.pdf")
print("Saved plot to target_prediction.pdf")

# --- Generate uniform samples from y_valid, keep as many as possible ---
num_bins = 20  # same as used in uniform_filter
bins = np.linspace(0, 1, num_bins + 1)  # define uniform bins
chosen_indices = []

# For each bin, collect all indices that fall in it
for i in range(num_bins):
	bin_mask = (y_valid >= bins[i]) & (y_valid < bins[i+1])
	bin_indices = np.where(bin_mask)[0]
	if len(bin_indices) > 0:
		# shuffle indices to randomize
		np.random.shuffle(bin_indices)
		chosen_indices.extend(bin_indices)

# Convert to array and optionally shuffle globally
chosen_indices = np.array(chosen_indices)
np.random.shuffle(chosen_indices)

print(f"Total chosen indices for uniform distribution: {len(chosen_indices)}")

uniform_samples = y_valid[chosen_indices]
# --- Plot histogram of uniform samples ---
plt.figure(figsize=(8,6))
plt.hist(uniform_samples.numpy(), bins=20, color='skyblue', edgecolor='black')
plt.xlabel("y_valid")
plt.ylabel("Frequency")
plt.title("Histogram of Uniformly Sampled y_valid Values")
plt.tight_layout()
plt.savefig("histogram.pdf")
plt.close()
