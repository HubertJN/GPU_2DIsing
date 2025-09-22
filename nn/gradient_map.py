import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from modules.architecture import CNN
from modules.dataset import to_cnn_dataset, load_hdf5_raw, filter_zero_committor
from scipy.ndimage import label
import os

# --- Config ---
h5_path = "training_gridstates.hdf5"
checkpoint_path = "models/cnn.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

folder = "figures"

save_saliency_path = os.path.join(folder, "cnn_saliency.pdf")
save_sigmoid_cluster = os.path.join(folder, "sigmoid_cluster.pdf")
save_pred_cnn = os.path.join(folder, "predicted_cnn.pdf")
save_pred_sigmoid = os.path.join(folder, "predicted_cluster.pdf")
figsize = (6, 6)  # Figure size for all plots
test_size = 0.2   # fraction for validation set

# --- Load dataset ---
grids, attrs = load_hdf5_raw(h5_path)
print("Filtering zero committor values...")
grids, attrs = filter_zero_committor(grids, attrs)

# --- Split into training and validation sets (indices) ---
num_samples = len(grids)
indices = np.arange(num_samples)
train_idx, valid_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

# --- Create train/validation datasets ---
train_ds = to_cnn_dataset(grids[train_idx], attrs[train_idx], device=device, augment_rotations=True)
valid_ds = to_cnn_dataset(grids[valid_idx], attrs[valid_idx], device=device, augment_rotations=False)

print(f"Train size: {len(train_ds)}, Validation size: {len(valid_ds)}")

# --- Load trained CNN model ---
model = CNN(input_size=64, channels=16, num_cnn_layers=4, num_fc_layers=1, dropout=0.1).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("CNN model loaded.")

# --- Saliency plot for first validation sample ---
x, y = valid_ds[0]
x_grid = x.squeeze(0).cpu().numpy()

# Identify largest cluster and roll to center
from scipy.ndimage import label
labeled_array, num_features = label(x_grid > 0)
cluster_sizes = [(labeled_array == i+1).sum() for i in range(num_features)]
largest_idx = np.argmax(cluster_sizes)
cluster_mask = (labeled_array == largest_idx + 1)
coords = np.argwhere(cluster_mask)
center_y, center_x = coords.mean(axis=0)
shift_y = int(x_grid.shape[0]//2 - center_y)
shift_x = int(x_grid.shape[1]//2 - center_x)
x_rolled = np.roll(x_grid, shift=(shift_y, shift_x), axis=(0,1))
x_tensor = torch.tensor(x_rolled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
x_tensor.requires_grad = True

# Forward/backward pass
output = model(x_tensor)
output.backward()
grad = x_tensor.grad.squeeze().cpu().numpy()

# Create RGBA gradient overlay: suppress very small gradients (<1%)
grad_abs = np.abs(grad)
grad_norm = grad_abs / grad_abs.max()

alpha = np.zeros_like(grad_norm)
mask = grad_norm > 0.02
alpha[mask] = 0.1 + 0.9 * (grad_norm[mask] - 0.01) / (1 - 0.01)

saliency_overlay = np.zeros((*grad.shape, 4))
saliency_overlay[..., 0] = 1.0  # red channel
saliency_overlay[..., 1] = 0.0  # green channel
saliency_overlay[..., 2] = 0.0  # blue channel
saliency_overlay[..., 3] = alpha  # alpha channel

# Prepare figure
fig, ax = plt.subplots(figsize=figsize)

# Mark up spins as X's using scatter
up_coords = np.argwhere(x_rolled > 0)
ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color='black', s=10)

# Overlay saliency gradient
ax.imshow(saliency_overlay, interpolation='none')

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Saliency Overlay", fontsize=12)
plt.tight_layout()
plt.savefig(save_saliency_path)
print(f"Saved saliency plot to {save_saliency_path}")
plt.close()

# ----------------- Sigmoid fit -----------------
# Use training dataset attributes
train_cluster_size = attrs[:,1][train_idx]
train_committor = attrs[:,2][train_idx]

def sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

# Fit sigmoid on training set
popt, _ = curve_fit(sigmoid, train_cluster_size, train_committor, p0=[0.1, np.median(train_committor)])

# --- Plot cluster size vs committor with sigmoid fit ---
x_fit = np.linspace(train_cluster_size.min(), train_cluster_size.max(), 200)
y_fit = sigmoid(x_fit, *popt)

plt.figure(figsize=figsize)
plt.scatter(train_cluster_size, train_committor, s=5, alpha=0.5, label="Training Data")
plt.plot(x_fit, y_fit, color='red', linewidth=2, label="Sigmoid Fit")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Target Committor")
plt.title("Cluster Size vs Committor with Sigmoid Fit")
plt.legend()
plt.tight_layout()
plt.savefig(save_sigmoid_cluster)
print(f"Saved cluster size vs committor plot to {save_sigmoid_cluster}")
plt.close()

# --- Predictions on validation set ---
y_valid = np.array([valid_ds[i][1].item() for i in range(len(valid_ds))])
x_valid = torch.stack([valid_ds[i][0] for i in range(len(valid_ds))]).to(device)

predictions = np.zeros(len(y_valid))
with torch.no_grad():
    for i, img in enumerate(x_valid):
        pred = model(img.unsqueeze(0)).item()
        predictions[i] = pred

# Compute RMSE
rmse = np.sqrt(np.mean((predictions - y_valid) ** 2))

# --- Plot predicted vs target committor for validation set ---
plt.figure(figsize=figsize)
plt.scatter(y_valid, predictions, s=5, alpha=0.5)
plt.plot([0,1],[0,1], color='red', linewidth=2)
plt.xlabel("Target Committor")
plt.ylabel("Predicted Committor")
plt.title("CNN Predictions on Validation Set")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left', fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(save_pred_cnn)
print(f"Saved CNN prediciton plot to {save_pred_cnn}")
plt.close()

# --- Plot Sigmoid Prediction ---
predictions = sigmoid(attrs[valid_idx][:,1], *popt)
# Compute RMSE
rmse = np.sqrt(np.mean((predictions - y_valid) ** 2))

plt.figure(figsize=figsize)
plt.scatter(y_valid, predictions, s=5, alpha=0.5)
plt.plot([0,1],[0,1], color='red', linewidth=2)
plt.xlabel("Target Committor")
plt.ylabel("Predicted Committor")
plt.title("Cluster Prediction on Validation Set")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left', fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(save_pred_sigmoid)
print(f"Saved sigmoid prediction plot to {save_pred_sigmoid}")
plt.close()