import yaml
import numpy as np
import h5py
import re
import time

class Dict2Obj:
  """Recursively turn dict into object with dot notation access."""
  def __init__(self, d):
    for k, v in d.items():
      if isinstance(v, dict):
        v = Dict2Obj(v)
      setattr(self, k, v)
  def __getitem__(self, key):
    return getattr(self, key)
  #def __getattr__(self, name):
  #  return getattr(self, name)

def load_config(path="config.yaml"):
  with open(path, "r") as f:
    cfg_dict = yaml.safe_load(f)
  return Dict2Obj(cfg_dict)

def read_attr(ds, name):
    val = ds.attrs.get(name)
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    if val == 'null':
        return np.nan
    return float(val)

# Function to compute magnetisation
def magnetisation(grid):
    return np.sum(grid)/len(grid)**2

# Index function
def index(isnap, igrid):
    grid_idx = isnap*ngrids + igrid
    return grid_idx

def write_training_hdf5(outpath, grids, attrs, L):
    """Append grids and attrs into outpath HDF5 file, creating it if missing.

    If the file exists, ensure header L matches, read current total_saved_grids,
    and append new datasets named grid_<n> with increasing indices.
    """
    # Open file for read/write/create
    with h5py.File(outpath, 'a') as fo:
        # If 'L' doesn't exist create it; otherwise ensure it matches
        if 'L' not in fo:
            fo.create_dataset('L', data=np.int32(L))
        else:
            existing_L = int(fo['L'][()])
            if existing_L != L:
                raise ValueError(f"Existing L={existing_L} does not match provided L={L}")

        # Determine starting index from total_saved_grids
        if 'total_saved_grids' in fo:
            start = int(fo['total_saved_grids'][()])
        else:
            fo.create_dataset('total_saved_grids', data=np.int32(0))
            start = 0

        names = ['magnetisation', 'lclus_size', 'committor', 'committor_error']
        for i in range(grids.shape[0]):
            idx = start + i
            dset = fo.create_dataset(f'grid_{idx}', data=grids[i].astype(np.int8))
            for j, name in enumerate(names):
                val = attrs[i, j]
                if np.isnan(val):
                    dset.attrs[name] = 'null'
                else:
                    dset.attrs[name] = float(val)

        # update total_saved_grids
        fo['total_saved_grids'][()] = np.int32(start + grids.shape[0])
        
# Function to compute magnetisation
def magnetisation(grid):
    return np.sum(grid)/len(grid)**2

def load_into_array(h5path):
    start = time.perf_counter()
    with h5py.File(h5path, "r") as f:
        total_saved = int(f['total_saved_grids'][()])
        L = int(f['L'][()])
        nbits = L * L
        nbytes = (nbits + 7) // 8

        # Read grids and attrs directly
        raw_grids = f['grids'][()]        # shape: (total_saved, nbytes)
        grids = np.empty((total_saved, L, L), dtype=np.int8)

        for i in range(total_saved):
            arr = np.frombuffer(raw_grids[i], dtype=np.uint8)
            bits = np.unpackbits(arr, bitorder='little')[:nbits]
            grids[i] = (bits.astype(np.int8) * 2 - 1).reshape(L, L)

        attrs = f['attrs'][()]             # shape: (total_saved, 4), doubles

    end = time.perf_counter()
    print(f"Loaded {len(grids)} datasets and attributes into memory.")
    print(f"Elapsed time: {end - start:.2f} s")
    return grids, attrs