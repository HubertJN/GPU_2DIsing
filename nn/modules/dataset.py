import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

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

# ----------------- CNN Dataset with Rotations + Reflections -----------------
class IsingDatasetCNN(Dataset):
  def __init__(self, grids, labels, device="cpu", augment=False):
    self.device = device

    grids = grids.to(device)
    labels = labels.to(device)

    augmented_grids = [grids]
    augmented_labels = [labels]

    if augment:
      # Add 90°, 180°, 270° rotations
      for k in [1, 2, 3]:
        rotated = torch.rot90(grids, k=k, dims=(2, 3))
        augmented_grids.append(rotated)
        augmented_labels.append(labels)

      # Collect reflections for each existing grid in augmented_grids
      new_grids = []
      new_labels = []
      for g, l in zip(augmented_grids, augmented_labels):
        # Horizontal flip (x-axis)
        new_grids.append(torch.flip(g, dims=[2]))
        new_labels.append(l)
        # Vertical flip (y-axis)
        new_grids.append(torch.flip(g, dims=[3]))
        new_labels.append(l)
        # Main diagonal (transpose x<->y)
        new_grids.append(g.transpose(2, 3))
        new_labels.append(l)
        # Anti-diagonal (flip + transpose)
        new_grids.append(torch.flip(g, dims=[2,3]).transpose(2, 3))
        new_labels.append(l)

      augmented_grids += new_grids
      augmented_labels += new_labels

    # Concatenate all augmented versions
    self.data = torch.cat(augmented_grids, dim=0)
    self.labels = torch.cat(augmented_labels, dim=0)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

def to_cnn_dataset(grids, attrs, device="cpu", augment=False):
  data_tensor = torch.tensor(grids, dtype=torch.float32).unsqueeze(1)
  labels_tensor = torch.tensor(attrs[:, 2], dtype=torch.float32).unsqueeze(1)
  return IsingDatasetCNN(data_tensor, labels_tensor, device=device, augment=augment)

# ----------------- GNN Dataset -----------------
class IsingDatasetGNN(Dataset):
  def __init__(self, grids, attrs, device="cpu"):
    self.device = device
    self.N, self.L, _ = grids.shape
    self.edge_index = self._compute_edges(self.L).to(device)

    self.graphs = []
    labels_tensor = torch.tensor(attrs[:, 2], dtype=torch.float32).to(device)
    for idx in range(self.N):
      x = torch.tensor(grids[idx], dtype=torch.float32).view(-1, 1).to(device)
      y = labels_tensor[idx].unsqueeze(0)
      self.graphs.append(Data(x=x, edge_index=self.edge_index, y=y))

  def _compute_edges(self, L):
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

# ----------------- Uniform distribution filter -----------------
def uniform_filter(data, labels, num_bins=10, seed=42):
  np.random.seed(seed)
  bins = np.linspace(0, 1, num_bins + 1)
  num_to_sample = 100  # len(data)
  subset_indices = []

  for i in range(num_bins):
    bin_idx = np.where((labels >= bins[i]) & (labels < bins[i+1]))[0]
    if len(bin_idx) == 0:
      continue
    num_to_sample = min(num_to_sample, len(bin_idx))

  for i in range(num_bins):
    bin_idx = np.where((labels >= bins[i]) & (labels < bins[i+1]))[0]
    selected = np.random.choice(bin_idx, size=num_to_sample, replace=False)
    subset_indices.extend(selected)

  return np.array(subset_indices)

# ----------------- Filter zero committor -----------------
def filter_zero_committor(grids, attrs):
  mask = attrs[:, 2] != 0
  grids_filtered = grids[mask]
  attrs_filtered = attrs[mask]
  print(f"Filtered {len(grids) - len(grids_filtered)} grids with committor=0")
  return grids_filtered, attrs_filtered

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
    valid_dl = TorchDataLoader(valid_ds, batch_size=1)
  else:  # gnn
    train_dl = PyGDataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = PyGDataLoader(valid_ds, batch_size=1)

  return train_dl, valid_dl

def prepare_subset(h5path, uniform_bins=10, test_size=0.2, seed=42):
  """
  Load grids and attributes from HDF5 and select a uniform subset.

  Returns:
      grids_subset: np.ndarray
      attrs_subset: np.ndarray
      subset_indices: np.ndarray
  """
  print("Loading full dataset...")
  grids, attrs = load_hdf5_raw(h5path)

  # np.random.seed(seed)  # ensure reproducibility
  # subset_indices = uniform_filter(grids, attrs[:,2], num_bins=uniform_bins)
  # grids_subset = grids[subset_indices]
  # attrs_subset = attrs[subset_indices]

  print("Filtering 0 committor data")
  grids, attrs = filter_zero_committor(grids, attrs)

  grids_subset = grids
  attrs_subset = attrs

  print(f"Uniform subset size: {len(grids_subset)}")

  # Split indices BEFORE augmentation
  num_samples = len(grids_subset)
  indices = np.arange(num_samples)
  train_idx, valid_idx = train_test_split(
    indices, test_size=test_size, random_state=seed, shuffle=True
  )

  return grids_subset, attrs_subset, train_idx, valid_idx

def prepare_datasets(grids, attrs, train_idx, valid_idx, model_type="cnn", device="cpu", batch_size=64, augment=True):
  """
  Take grids and attributes with train/valid indices, and produce datasets and dataloaders.

  Returns:
      train_dl, valid_dl, train_ds, valid_ds
  """
  # Create train/validation datasets
  if model_type.lower() == "cnn":
    train_ds = to_cnn_dataset(grids[train_idx], attrs[train_idx],
                              device=device, augment=augment)
    valid_ds = to_cnn_dataset(grids[valid_idx], attrs[valid_idx],
                              device=device, augment=False)
  else:
    train_ds = to_gnn_dataset(grids[train_idx], attrs[train_idx], device=device)
    valid_ds = to_gnn_dataset(grids[valid_idx], attrs[valid_idx], device=device)

  print(f"Train size: {len(train_ds)}, Validation size: {len(valid_ds)}")

  # Create DataLoaders
  train_dl, valid_dl = data_to_dataloader(train_ds, valid_ds, batch_size,
                                          model_type=model_type.lower())
  print(f"DataLoaders created: train_batches={len(train_dl)}, valid_batches={len(valid_dl)}")

  return train_dl, valid_dl, train_ds, valid_ds
