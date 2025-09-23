import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from modules.architecture import CNN
from modules.dataset import prepare_subset, prepare_datasets
from modules.config import load_config
from scipy.ndimage import label

# --- Load config ---
config = load_config("config.yaml")

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset parameters ---
h5path = config.paths.data
test_size = config.dataset.test_size
batch_size = config.dataset.batch_size
model_type = config.model.type.lower()

# --- Load full dataset ---
grids, attrs, train_idx, valid_idx = prepare_subset(h5path, test_size=test_size)
train_dl, valid_dl, train_ds, valid_ds = prepare_datasets(
  grids, attrs, train_idx, valid_idx,
  model_type, device,
  batch_size,
  augment=config.dataset.augment
)

# --- Load trained CNN model ---
model = CNN(
  input_size=config.model.input_size,
  channels=config.model.channels,
  num_cnn_layers=config.model.num_cnn_layers,
  num_fc_layers=config.model.num_fc_layers,
  dropout=config.model.dropout
).to(device)
model.load_state_dict(torch.load(f"{config.paths.save_dir}/{config.model.type}.pth", map_location=device))
model.eval()
print(f"{config.model.type.upper()} model loaded.")

# --- Saliency plot for first validation sample ---
x, y = valid_ds[0]
x_grid = x.squeeze(0).cpu().numpy()

# Identify largest cluster and roll to center
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

# Create RGBA gradient overlay
grad_abs = np.abs(grad)
grad_norm = grad_abs / grad_abs.sum()
uniform_min = 1 / grad_norm.size
uniform_factor = 1.25 * uniform_min
alpha = np.zeros_like(grad_norm)
mask = grad_norm > uniform_factor
alpha[mask] = 0.1 + 0.9 * (grad_norm[mask] - uniform_factor) / (grad_norm[mask].max() - uniform_factor)

saliency_overlay = np.zeros((*grad.shape, 4))
saliency_overlay[..., 0] = 1.0
saliency_overlay[..., 1] = 0.0
saliency_overlay[..., 2] = 0.0
saliency_overlay[..., 3] = alpha

# Plot saliency
fig, ax = plt.subplots(figsize=(6, 6))
up_coords = np.argwhere(x_rolled > 0)
ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color='black', s=10)
ax.imshow(saliency_overlay, interpolation='bicubic')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Saliency Overlay", fontsize=12)
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/{config.model.type}_saliency.pdf")
print(f"Saved saliency plot to {config.paths.plot_dir}/{config.model.type}_saliency.pdf")
plt.close()

# --- Sigmoid fit ---
train_cluster_size = attrs[:,1][train_idx]
train_committor = attrs[:,2][train_idx]

def sigmoid(x, k, x0):
  return 1 / (1 + np.exp(-k * (x - x0)))

popt, _ = curve_fit(sigmoid, train_cluster_size, train_committor,
                    p0=[0.1, np.median(train_committor)], maxfev=10000)

x_fit = np.linspace(train_cluster_size.min(), train_cluster_size.max(), 200)
y_fit = sigmoid(x_fit, *popt)

plt.figure(figsize=(6,6))
plt.scatter(train_cluster_size, train_committor, s=5, alpha=0.5, label="Training Data")
plt.plot(x_fit, y_fit, color='red', linewidth=2, label="Sigmoid Fit")
plt.vlines(7, 0, 1, linestyle="dashed", color="grey")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Target Committor")
plt.title("Cluster Size vs Committor with Sigmoid Fit")
plt.legend()
plt.tight_layout()
plt.xlim(-25, 300)
plt.savefig(f"{config.paths.plot_dir}/sigmoid_cluster.pdf")
print(f"Saved cluster size vs committor plot to {config.paths.plot_dir}/sigmoid_cluster.pdf")
plt.close()

# --- CNN Predictions on validation set ---
y_valid = np.array([valid_ds[i][1].item() for i in range(len(valid_ds))])
x_valid = torch.stack([valid_ds[i][0] for i in range(len(valid_ds))]).to(device)

with torch.no_grad():
  predictions = model(x_valid).squeeze().cpu().numpy()

rmse = np.sqrt(np.mean((predictions - y_valid) ** 2))

plt.figure(figsize=(6,6))
plt.scatter(y_valid, predictions, s=5, alpha=0.5)
plt.plot([0,1],[0,1], color='red', linewidth=2)
plt.xlabel("Target Committor")
plt.ylabel("Predicted Committor")
plt.title("CNN Predictions on Validation Set")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left', fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/predicted_cnn.pdf")
print(f"Saved CNN prediction plot to {config.paths.plot_dir}/predicted_cnn.pdf")
plt.close()

# --- Sigmoid Prediction ---
predictions = sigmoid(attrs[valid_idx][:,1], *popt)
rmse = np.sqrt(np.mean((predictions - y_valid) ** 2))

plt.figure(figsize=(6,6))
plt.scatter(y_valid, predictions, s=5, alpha=0.5)
plt.plot([0,1],[0,1], color='red', linewidth=2)
plt.xlabel("Target Committor")
plt.ylabel("Predicted Committor")
plt.title("Cluster Prediction on Validation Set")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left', fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/predicted_cluster.pdf")
print(f"Saved sigmoid prediction plot to {config.paths.plot_dir}/predicted_cluster.pdf")
plt.close()
