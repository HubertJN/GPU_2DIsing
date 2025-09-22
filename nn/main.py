import argparse
import sys
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from modules.architecture import CNN, GNN, data_to_dataloader, fit, loss_batch, physics_func
from modules.dataset import load_hdf5_raw, to_cnn_dataset, to_gnn_dataset, uniform_filter, filter_zero_committor
import numpy as np
import matplotlib.pyplot as plt

# --- Command-line arguments ---
parser = argparse.ArgumentParser(description="Train CNN or GNN on 2D Ising grids")
parser.add_argument("--model", type=str, choices=["cnn", "gnn"], required=True,
                    help="Choose model type: cnn or gnn (required)")
args = parser.parse_args()

model_type = args.model.lower()
if model_type not in ["cnn", "gnn"]:
    print(f"Invalid model type '{args.model}'. Exiting.")
    sys.exit(1)

# --- Path Config ---
h5path = "../nn/training_gridstates.hdf5"
save_path = f"models/{model_type}.pth"

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 64
lr = 0.001
wd = 0.0001
test_size = 0.2  # fraction for validation

# --- Load full dataset ---
print("Loading full dataset...")
grids, attrs = load_hdf5_raw(h5path)

print("Filtering for uniform committor distribution...")
subset_indices = uniform_filter(grids, attrs[:,2], num_bins=10)
grids = grids[subset_indices]
attrs = attrs[subset_indices]

print("Filtering zero committor values...")
grids, attrs = filter_zero_committor(grids, attrs)

# --- Split indices BEFORE augmentation ---
num_samples = len(grids)
indices = np.arange(num_samples)
train_idx, valid_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

# --- Create train/validation datasets ---
if model_type == "cnn":
    train_ds = to_cnn_dataset(grids[train_idx], attrs[train_idx], device=device, augment_rotations=True)
    valid_ds = to_cnn_dataset(grids[valid_idx], attrs[valid_idx], device=device, augment_rotations=True)
else:
    train_ds = to_gnn_dataset(grids[train_idx], attrs[train_idx], device=device)
    valid_ds = to_gnn_dataset(grids[valid_idx], attrs[valid_idx], device=device)

print(f"Train size: {len(train_ds)}, Validation size: {len(valid_ds)}")

# --- Create DataLoaders ---
train_dl, valid_dl = data_to_dataloader(train_ds, valid_ds, batch_size, model_type=model_type)
print(f"DataLoaders created: train_batches={len(train_dl)}, valid_batches={len(valid_dl)}")

# --- Initialize model, loss, optimizer ---
if model_type == "cnn":
    model = CNN(input_size=64, channels=16, num_cnn_layers=4, num_fc_layers=1, dropout=0.1).to(device)
else:
    model = GNN().to(device)

# --- Remove weight decay from BN/bias ---
decay, no_decay = [], []
for name, param in model.named_parameters():
	if "bn" in name or "bias" in name:
		no_decay.append(param)
	else:
		decay.append(param)

optimizer = torch.optim.AdamW([
	{'params': decay, 'weight_decay': wd},
	{'params': no_decay, 'weight_decay': 0.0}
], lr=lr)

loss_func = torch.nn.SmoothL1Loss()

# --- Train ---
print(f"Starting training for {model_type.upper()}...")
fit(epochs, model, loss_func, physics_func, optimizer, train_dl, valid_dl, device)
print("Training complete.")

# --- Save model weights ---
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

# --- Run predictions on the validation set ---
model.eval()
predictions = []
y_valid = []

with torch.no_grad():
    for i in range(len(valid_ds)):
        x, y = valid_ds[i]
        x = x.unsqueeze(0).to(device)  # add batch dimension
        pred = model(x).squeeze().item()
        predictions.append(pred)
        y_valid.append(y.item())

predictions = np.array(predictions)
y_valid = np.array(y_valid)

# Accuracy and RMSE
correct = np.sum(np.abs(predictions - y_valid) < 0.05)
accuracy = 100 * correct / len(y_valid)
rmse = np.sqrt(np.mean((predictions - y_valid) ** 2))

print("============================================")
print(f"Accuracy for validation set : {accuracy:.2f} %")
print("Number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("RMSE: ", rmse)
print("============================================")

# --- Scatter plot ---
plt.figure(figsize=(5, 5))
plt.scatter(y_valid, predictions, s=0.5)
plt.plot([0, 1], [0, 1], c="red")
plt.xlim(-0.1, 1.1)
plt.xlabel("Target")
plt.ylabel("Prediction")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left', fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.savefig("valid_target_prediction.pdf")
print("Saved plot to valid_target_prediction.pdf")

# --- Spatial invariance test on a random validation image ---
with torch.no_grad():
    iimg = np.random.randint(len(valid_ds))
    base_img, label = valid_ds[iimg]
    base_img = base_img.unsqueeze(0).to(device)

    out_orig = model(base_img).squeeze().item()
    shifted_img = torch.roll(base_img, shifts=(0, 16), dims=(2, 3))
    out_shift = model(shifted_img).squeeze().item()
    rotated_img = torch.rot90(base_img, k=1, dims=(2, 3))
    out_rot = model(rotated_img).squeeze().item()
    combined_img = torch.roll(rotated_img, shifts=(0, 16), dims=(2, 3))
    out_comb = model(combined_img).squeeze().item()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    images = [base_img, shifted_img, rotated_img, combined_img]
    images = [img.squeeze(0).squeeze(0).cpu() for img in images]
    outputs = [out_orig, out_shift, out_rot, out_comb]
    titles = ["Original", "Translated", "Rotated", "Combined"]

    for ax, img, title, out_val in zip(axs, images, titles, outputs):
        ax.imshow(img.numpy(), cmap="gray")
        diff = out_val - out_orig
        ax.set_title(title)
        ax.set_xlabel(f"Pred: {out_val:.3f}\nDiff: {diff:+.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("valid_invariance_test.pdf")
    print("Saved invariance test plot to valid_invariance_test.pdf")
