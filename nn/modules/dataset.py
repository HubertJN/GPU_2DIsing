import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# ----------------- Raw HDF5 loading -----------------
def load_hdf5_raw(h5path):
    """
    Load grids and attributes from HDF5 file as NumPy arrays.
    Returns:
        grids: np.ndarray, shape (N, L, L)
        attrs: np.ndarray, shape (N, 4)
    """
    def read_attr(ds, name):
        val = ds.attrs.get(name)
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        if val == 'null':
            return np.nan
        return float(val)

    print(f"Opening HDF5 file: {h5path}")
    with h5py.File(h5path, 'r') as f:
        total_saved = int(f['total_saved_grids'][()])
        L = int(f['L'][()])
        grids = np.empty((total_saved, L, L), dtype=np.int8)
        attrs = np.empty((total_saved, 4), dtype=np.float64)

        idx = 0
        for grid_name in sorted(k for k in f.keys() if k.startswith('grid_')):
            grp = f[grid_name]
            if idx >= total_saved:
                break

            grids[idx] = grp[()]
            attrs[idx, 0] = read_attr(grp, 'magnetisation')
            attrs[idx, 1] = read_attr(grp, 'lclus_size')
            attrs[idx, 2] = read_attr(grp, 'committor')
            attrs[idx, 3] = read_attr(grp, 'committor_error')

            idx += 1

    print(f"Loaded {idx} grids. grids.shape={grids.shape}, attrs.shape={attrs.shape}")
    return grids, attrs


# ----------------- CNN Dataset -----------------
class IsingDatasetCNN(Dataset):
    def __init__(self, grids, labels, device="cpu"):
        self.data = grids.to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def to_cnn_dataset(grids, attrs, device="cpu"):
    data_tensor = torch.tensor(grids, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(attrs[:, 2], dtype=torch.float32)
    return IsingDatasetCNN(data_tensor, labels_tensor, device=device)


# ----------------- GNN Dataset -----------------
class IsingDatasetGNN(Dataset):
    def __init__(self, grids, attrs, device="cpu"):
        """
        Converts 2D Ising grids into a PyG GNN dataset with periodic edges,
        styled similarly to a simple CNN dataset class.
        """
        self.device = device
        self.N, self.L, _ = grids.shape
        self.edge_index = self._compute_edges(self.L).to(device)

        # Convert grids and labels to tensors
        self.graphs = []
        labels_tensor = torch.tensor(attrs[:, 2], dtype=torch.float32).to(device)
        for idx in range(self.N):
            x = torch.tensor(grids[idx], dtype=torch.float32).view(-1, 1).to(device)
            y = labels_tensor[idx].unsqueeze(0)
            self.graphs.append(Data(x=x, edge_index=self.edge_index, y=y))

    def _compute_edges(self, L):
        """Preallocate edge tensor for L x L grid with periodic boundaries."""
        num_nodes = L * L
        num_edges = num_nodes * 4
        edges = torch.empty((2, num_edges), dtype=torch.long)

        idx = 0
        for i in range(L):
            for j in range(L):
                node = i * L + j
                neighbors = [
                    ((i - 1) % L) * L + j,  # up
                    ((i + 1) % L) * L + j,  # down
                    i * L + (j - 1) % L,    # left
                    i * L + (j + 1) % L     # right
                ]
                for n in neighbors:
                    edges[0, idx] = node
                    edges[1, idx] = n
                    idx += 1
        return edges.contiguous()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx]

def to_gnn_dataset(grids, attrs, device="cpu"):
    return IsingDatasetGNN(grids, attrs, device=device)