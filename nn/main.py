import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from modules.architecture import CNN, data_to_dataloader, fit, loss_batch
from modules.dataset import load_hdf5_as_dataset

# --- Path Config ---
h5_path = "training_gridstates.hdf5"  # single HDF5 file
save_path = "models/cnn.pth"

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 10
lr = 1e-3
test_size = 0.2  # fraction for validation

# --- Load full dataset ---
print("Loading full dataset...")
full_ds = load_hdf5_as_dataset(h5_path, device=device)
print(f"Full dataset size: {len(full_ds)}")

# --- Split indices ---
indices = list(range(len(full_ds)))
train_idx, valid_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

train_ds = Subset(full_ds, train_idx)
valid_ds = Subset(full_ds, valid_idx)
print(f"Train size: {len(train_ds)}, Validation size: {len(valid_ds)}")

# --- Create DataLoaders ---
train_dl, valid_dl = data_to_dataloader(train_ds, valid_ds, batch_size)
print(f"DataLoaders created: train_batches={len(train_dl)}, valid_batches={len(valid_dl)}")

exit()
# --- Initialize model, loss, optimizer ---
model = CNN(input_size=64).to(device)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Train ---
print("Starting training...")
fit(epochs, model, loss_func, optimizer, train_dl, valid_dl)

print("Training complete.")

# --- Save model weights ---
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")