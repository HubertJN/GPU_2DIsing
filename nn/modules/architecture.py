import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

def loss_batch(model, loss_func, physics_loss, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    loss += physics_loss
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def physics_func(model, xb, one, zero):
    loss = 0
    #loss += 0.1 * (model(zero).mean()) ** 2
    #loss += 0.1 * ((model(one) - 1).mean()) ** 2
    return loss

def fit(epochs, model, loss_func, physics_func, opt, train_dl, valid_dl, device="cpu"):
    zero = torch.zeros([64,64], dtype=torch.float32).to(device)
    one = torch.ones([64,64], dtype=torch.float32).to(device)

    # Determine width for epoch numbers (4 digits max)
    width = max(4, len(str(epochs)))

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            loss_val, n = loss_batch(model, loss_func, physics_func(model, xb, one, zero), xb, yb, opt)
            train_losses.append((loss_val, n))

        # Weighted average of training loss
        train_vals, train_counts = zip(*train_losses)
        train_loss = np.sum(np.multiply(train_vals, train_counts)) / np.sum(train_counts)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_losses = [loss_batch(model, loss_func, physics_func(model, xb, one, zero), xb, yb) for xb, yb in valid_dl]
            val_vals, val_counts = zip(*val_losses)
            val_loss = np.sum(np.multiply(val_vals, val_counts)) / np.sum(val_counts)

        # Print with fixed-width epoch formatting
        print(f"Epoch {epoch+1:{width}}/{epochs:{width}} - Train Loss: {train_loss:.5f} - Validation Loss: {val_loss:.5f}")

def data_to_dataloader(train_ds, valid_ds, bs, model_type="cnn"):
    """
    Return DataLoaders for training and validation datasets.
    
    Args:
        train_ds, valid_ds: Dataset objects
        bs: batch size
        model_type: "cnn" or "gnn"
    """
    if model_type == "cnn":
        train_dl = TorchDataLoader(train_ds, batch_size=bs, shuffle=True)
        valid_dl = TorchDataLoader(valid_ds, batch_size=bs)
    else:  # gnn
        train_dl = PyGDataLoader(train_ds, batch_size=bs, shuffle=True)
        valid_dl = PyGDataLoader(valid_ds, batch_size=bs)
    
    return train_dl, valid_dl

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=1, padding=padding,
                              dilation=dilation, padding_mode="circular")
        self.dropout = nn.Dropout(dropout)
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.conv(x))
        out = self.dropout(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out += identity
        return F.leaky_relu(out)

class CNN(nn.Module):
    def __init__(self, input_size=64, channels=16, dropout=0.2,
                 num_cnn_layers=3, num_fc_layers=2):
        super().__init__()
        self.input_size = input_size

        # Build CNN layers
        self.cnn_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            in_ch = 1 if i == 0 else channels
            dilation = 1 if i == 0 else (2**i)  # example progressive dilation
            self.cnn_layers.append(ResidualCNN(in_ch, channels, dilation=dilation, dropout=dropout))

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Build fully connected residual layers
        self.fc_layers = nn.ModuleList()
        for _ in range(num_fc_layers):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout)
                )
            )

        self.out = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.input_size, self.input_size)
        for layer in self.cnn_layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Apply fully connected residual layers
        for fc in self.fc_layers:
            identity = x
            x = identity + fc(x)

        x = self.out(x)
        return x

class GNN(torch.nn.Module):
    """
    Simple GCN for 2D Ising grids.
    Treat each spin as a node; connect to 4 nearest neighbors (up/down/left/right).
    """
    def __init__(self, channels=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(1, channels)
        self.conv2 = GCNConv(channels, channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(channels, 1)

    def forward(self, data_batch):
        # data_batch: PyG Batch object with x (node features), edge_index, batch
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch
        x = x + F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = x + F.leaky_relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)  # aggregate node features per graph
        return torch.sigmoid(self.fc(x))