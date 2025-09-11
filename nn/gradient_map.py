import argparse
import sys
import torch
import matplotlib.pyplot as plt
from modules.architecture import CNN, GNN
from modules.dataset import to_cnn_dataset, to_gnn_dataset, load_hdf5_as_dataset

# --- Command-line argument ---
parser = argparse.ArgumentParser(description="Visualize CNN or GNN predictions on 2D Ising grids")
parser.add_argument("--model", type=str, choices=["cnn", "gnn"], required=True,
                    help="Choose model type: cnn or gnn (required)")
args = parser.parse_args()

model_type = args.model.lower()
if model_type not in ["cnn", "gnn"]:
    print(f"Invalid model type '{args.model}'. Exiting.")
    sys.exit(1)

# --- Config ---
h5_path = "training_gridstates.hdf5"
checkpoint_path = f"models/{model_type}.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
save_fig_path = f"{model_type}_saliency.pdf"  # output figure path

# --- Load dataset ---
grids, attrs = load_hdf5_as_dataset(h5_path, device="cpu")
if model_type == "cnn":
    dataset = to_cnn_dataset(grids, attrs, device=device)
else:
    dataset = to_gnn_dataset(grids, attrs, device=device)

print(f"Dataset size: {len(dataset)}")

# --- Load trained model ---
if model_type == "cnn":
    model = CNN(input_size=grids.shape[1]).to(device)
else:
    model = GNN().to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"{model_type.upper()} model loaded.")

# --- Select first sample ---
if model_type == "cnn":
    x, y = dataset[0]
    x = x.unsqueeze(0)  # add batch dimension
    x.requires_grad = True

    # Forward pass
    output = model(x)
    print(f"Model output: {output.item():.4f}")

    # Backward pass
    output.backward()
    grad = x.grad.squeeze().cpu()

    # Visualization
    plt.figure(figsize=(6,6))
    plt.imshow(x.detach().cpu().squeeze(), cmap='gray', interpolation='none')
    plt.imshow(grad, cmap='seismic', alpha=0.5, interpolation='none')
    plt.colorbar(label='Gradient')
    plt.title("CNN Input Grid with Saliency Overlay")
    plt.axis('off')
    plt.savefig(save_fig_path)
    print(f"Figure saved to {save_fig_path}")
    plt.close()

else:  # GNN
    data = dataset[0]
    L = int(data.x.size(0) ** 0.5)  # grid size
    x = data.x.clone().detach().requires_grad_(True)
    data.x = x

    # Forward pass
    output = model(data)
    print(f"Model output: {output.item():.4f}")

    # Backward pass
    output.backward()
    grad = data.x.grad.view(L, L).cpu()

    # Reshape input to grid
    x_grid = x.view(L, L).cpu()

    # Visualization
    plt.figure(figsize=(6,6))
    plt.imshow(x_grid, cmap='gray', interpolation='none')
    plt.imshow(grad, cmap='seismic', alpha=0.5, interpolation='none')
    plt.colorbar(label='Gradient')
    plt.title("GNN Input Grid with Saliency Overlay")
    plt.axis('off')
    plt.savefig(save_fig_path)
    print(f"Figure saved to {save_fig_path}")
    plt.close()
