import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.architecture import CNN, GNN, fit, physics_func
from modules.dataset import prepare_subset, prepare_datasets
from modules.config import load_config

# --- Load Config ---
config = load_config("config.yaml")

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load dataset ---
grids, attrs, train_idx, valid_idx = prepare_subset(
  config.paths.data, 
  test_size=config.dataset.test_size
)
train_dl, valid_dl, train_ds, valid_ds = prepare_datasets(
  grids, attrs, train_idx, valid_idx,
  config.model.type, device,
  config.dataset.batch_size,
  augment=config.dataset.augment
)

# --- Initialize model ---
if config.model.type == "cnn":
  model = CNN(
    input_size=config.model.input_size,
    channels=config.model.channels,
    num_cnn_layers=config.model.num_cnn_layers,
    num_fc_layers=config.model.num_fc_layers,
    dropout=config.model.dropout
  ).to(device)
else:
  model = GNN().to(device)

# --- Optimizer (separate BN/bias) ---
decay, no_decay = [], []
for name, param in model.named_parameters():
  if "bn" in name or "bias" in name:
    no_decay.append(param)
  else:
    decay.append(param)

optimizer = torch.optim.AdamW([
  {'params': decay, 'weight_decay': config.training.weight_decay},
  {'params': no_decay, 'weight_decay': 0.0}
], lr=config.training.lr)

# --- Loss ---
if config.training.loss.lower() == "smoothl1loss":
  loss_func = torch.nn.SmoothL1Loss()
elif config.training.loss.lower() == "mse":
  loss_func = torch.nn.MSELoss()
else:
  raise ValueError(f"Unsupported loss: {config.training.loss}")

# --- Train ---
print(f"Starting training for {config.model.type.upper()}...")
fit(config.training.epochs, model, loss_func, physics_func, optimizer,
    train_dl, valid_dl, device)
print("Training complete.")

# --- Save model ---
save_path = f"{config.paths.save_dir}/{config.model.type}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

# --- Validation Predictions ---
model.eval()
predictions, y_valid = [], []
with torch.no_grad():
  for i in range(len(valid_ds)):
    x, y = valid_ds[i]
    x = x.unsqueeze(0).to(device)
    pred = model(x).squeeze().item()
    predictions.append(pred)
    y_valid.append(y.item())

predictions = np.array(predictions)
y_valid = np.array(y_valid)

# --- Accuracy + RMSE ---
threshold = getattr(config.evaluation, "threshold", 0.05)
correct = np.sum(np.abs(predictions - y_valid) < threshold)
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
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='left',
         fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.savefig(f"{config.paths.plot_dir}/valid_target_prediction.pdf")
print("Saved plot to valid_target_prediction.pdf")

# --- Spatial invariance test ---
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

  plt.savefig(f"{config.paths.plot_dir}/valid_invariance_test.pdf")
  print("Saved invariance test plot to valid_invariance_test.pdf")
