import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    loss += 0.1 * (model(torch.zeros_like(xb)).mean()) ** 2
    loss += 0.1 * ((model(torch.ones_like(xb)) - 1).mean()) ** 2
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()

        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")


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


class CNN(torch.nn.Module):
    def __init__(self, input_size=64, channels=10):
        super().__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(1, channels, 3, stride=1, padding=1, dilation=1, padding_mode='circular')
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, stride=1, padding=4, dilation=4, padding_mode='circular')
        self.conv3 = torch.nn.Conv2d(channels, channels, 3, stride=1, padding=7, dilation=7, padding_mode='circular')
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(channels, 1)

    def forward(self, xb):
        xb = xb.view(-1, 1, self.input_size, self.input_size)
        xb = xb + torch.relu(self.conv1(xb))
        xb = xb + torch.relu(self.conv2(xb))
        xb = xb + torch.relu(self.conv3(xb))
        xb = self.pool(xb)
        xb = xb.view(xb.size(0), -1)
        return torch.sigmoid(self.fc(xb))

class GNN(torch.nn.Module):
    """
    Simple GCN for 2D Ising grids.
    Treat each spin as a node; connect to 4 nearest neighbors (up/down/left/right).
    """
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = GCNConv(1, channels)
        self.conv2 = GCNConv(channels, channels)
        self.fc = torch.nn.Linear(channels, 1)

    def forward(self, data_batch):
        # data_batch: PyG Batch object with x (node features), edge_index, batch
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch
        x = x + torch.relu(self.conv1(x, edge_index))
        x = x + torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # aggregate node features per graph
        return torch.sigmoid(self.fc(x))