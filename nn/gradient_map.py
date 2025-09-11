import torch
import matplotlib.pyplot as plt
from modules.architecture import CNN
from modules.dataset import load_hdf5_as_dataset

# --- Config ---
h5_path = "training_gridstates.hdf5"
checkpoint_path = "models/cnn.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load dataset ---
dataset = load_hdf5_as_dataset(h5_path, device=device)
print(f"Dataset size: {len(dataset)}")

# --- Load trained model ---
model = CNN(input_size=64).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Model loaded.")

# --- Select a sample from dataset ---
x, y = dataset[0]               # pick first grid
x = x.unsqueeze(0)              # add batch dimension
x.requires_grad = True          # track gradients w.r.t input

# --- Forward pass ---
output = model(x)
print(f"Model output: {output.item():.4f}")

# --- Backward pass w.r.t input ---
output.backward()
grad = x.grad.squeeze().cpu()   # remove batch/channel dims

# --- Visualization ---
plt.figure(figsize=(6,6))

# Display input grid in black and white
plt.imshow(x.detach().cpu().squeeze(), cmap='gray', interpolation='none')

# Overlay gradients (saliency) as transparent heatmap
plt.imshow(grad, cmap='seismic', alpha=0.5, interpolation='none')
plt.colorbar(label='Gradient')
plt.title("Input Grid with Saliency Overlay")
plt.axis('off')
plt.show()
