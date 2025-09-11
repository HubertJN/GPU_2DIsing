import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def read_attr(ds, name):
    """Safely read an attribute from an HDF5 dataset."""
    val = ds.attrs.get(name)
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    if val == 'null':
        return np.nan
    return float(val)


class IsingDataset(Dataset):
    def __init__(self, data, labels, device="cpu"):
        """
        Initializes Dataset object for Ising data.
        
        Args:
            data (torch.Tensor): Ising grids
            labels (torch.Tensor): committor values for each grid
            device (str): device to load tensors onto (default: "cpu")
        """
        self.data = data.to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def load_hdf5_as_dataset(h5path, device="cpu"):
    """
    Loads an HDF5 file and returns a PyTorch Dataset.

    Args:
        h5path (str): Path to HDF5 file.
        device (str): Device to store tensors ("cpu" or "cuda").

    Returns:
        IsingDataset: dataset wrapping grids and committor values.
    """
    print(f"Opening HDF5 file: {h5path}")
    with h5py.File(h5path, 'r') as f:
        total_saved = int(f['total_saved_grids'][()])
        L = int(f['L'][()])

        print(f"Header read: total_saved_grids={total_saved}, L={L}")

        training_grids = np.empty((total_saved, L, L), dtype=np.int8)
        training_attrs = np.empty((total_saved, 4), dtype=np.float64)

        idx = 0
        for grid_name in sorted(k for k in f.keys() if k.startswith('grid_')):
            grp = f[grid_name]
            if idx >= total_saved:
                break

            training_grids[idx] = grp[()]

            training_attrs[idx, 0] = read_attr(grp, 'magnetisation')
            training_attrs[idx, 1] = read_attr(grp, 'lclus_size')
            training_attrs[idx, 2] = read_attr(grp, 'committor')
            training_attrs[idx, 3] = read_attr(grp, 'committor_error')

            idx += 1

        print(f"Loaded {idx} grids. "
              f"grids.shape={training_grids.shape}, "
              f"attrs.shape={training_attrs.shape}")

    # Convert numpy arrays to tensors
    data_tensor = torch.tensor(training_grids, dtype=torch.float32).unsqueeze(1)  # add channel dim
    labels_tensor = torch.tensor(training_attrs[:, 2], dtype=torch.float32)       # committor only

    print("Converted numpy arrays to torch.Tensors")

    dataset = IsingDataset(data_tensor, labels_tensor, device=device)
    print("Dataset successfully created")

    return dataset
