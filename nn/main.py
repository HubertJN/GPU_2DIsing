import argparse
import sys
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from modules.architecture import CNN, GNN, data_to_dataloader, fit, loss_batch
from modules.dataset import load_hdf5_raw, to_cnn_dataset, to_gnn_dataset

# --- Command-line arguments ---
parser = argparse.ArgumentParser(description="Train CNN or GNN on 2D Ising grids")
parser.add_argument("--model", type=str, choices=["cnn", "gnn"], required=True,
                    help="Choose model type: cnn or gnn (required)")
args = parser.parse_args()

# Validate input (argparse already ensures choice, so this is just extra safety)
model_type = args.model.lower()
if model_type not in ["cnn", "gnn"]:
    print(f"Invalid model type '{args.model}'. Exiting.")
    sys.exit(1)

# --- Path Config ---
h5path = "training_gridstates.hdf5"
save_path = f"models/{model_type}.pth"

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 10
lr = 1e-3
test_size = 0.2  # fraction for validation

# --- Load full dataset ---
print("Loading full dataset...")
grids, attrs = load_hdf5_raw(h5path)

# Choose dataset type based on model
if model_type == "cnn":
    dataset = to_cnn_dataset(grids, attrs, device="cpu")
else:
    dataset = to_gnn_dataset(grids, attrs, device="cpu")

print(f"Full dataset size: {len(dataset)}")

# --- Split indices ---
indices = list(range(len(dataset)))
train_idx, valid_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

train_ds = Subset(dataset, train_idx)
valid_ds = Subset(dataset, valid_idx)
print(f"Train size: {len(train_ds)}, Validation size: {len(valid_ds)}")

# --- Create DataLoaders ---
train_dl, valid_dl = data_to_dataloader(train_ds, valid_ds, batch_size, model_type=model_type)
print(f"DataLoaders created: train_batches={len(train_dl)}, valid_batches={len(valid_dl)}")

# --- Initialize model, loss, optimizer ---
if model_type == "cnn":
    model = CNN(input_size=grids.shape[1]).to(device)
else:
    model = GNN().to(device)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

exit()
# --- Train ---
print(f"Starting training for {model_type.upper()}...")
fit(epochs, model, loss_func, optimizer, train_dl, valid_dl)
print("Training complete.")

# --- Save model weights ---
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")
