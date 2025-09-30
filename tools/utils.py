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
def index(isnap, igrid, ngrids):
    grid_idx = isnap*ngrids + igrid
    return grid_idx
       
# Function to compute magnetisation
def magnetisation(grid):
    return np.sum(grid)/len(grid)**2

def load_into_array(h5path, load_grids=True, indices=None):
    start = time.perf_counter()
    with h5py.File(h5path, "r") as f:
        total_saved = int(f['total_saved_grids'][()]) if 'total_saved_grids' in f else 0
        L = int(f['L'][()]) if 'L' in f else 0
        nbits = L * L
        nbytes = (nbits + 7) // 8

        if load_grids and 'grids' in f:
            if indices is None:
                raw_grids = f['grids'][()]
                grids = np.empty((total_saved, L, L), dtype=np.int8)
                for i in range(total_saved):
                    arr = np.frombuffer(raw_grids[i], dtype=np.uint8)
                    bits = np.unpackbits(arr, bitorder='little')[:nbits]
                    grids[i] = (bits.astype(np.int8) * 2 - 1).reshape(L, L)
            else:
                grids = np.empty((len(indices), L, L), dtype=np.int8)
                for i, idx in enumerate(indices):
                    arr = np.frombuffer(f['grids'][idx], dtype=np.uint8)
                    bits = np.unpackbits(arr, bitorder='little')[:nbits]
                    grids[i] = (bits.astype(np.int8) * 2 - 1).reshape(L, L)
        else:
            grids = np.empty((0, L, L), dtype=np.int8)

        header_keys = [key for key in f.keys() if key not in ('grids', 'attrs')]
        headers = {key: f[key][()] for key in header_keys}

        if indices is None:
            attrs = f['attrs'][()] if 'attrs' in f else np.empty((total_saved, 0))
        else:
            attrs = f['attrs'][indices] if 'attrs' in f else np.empty((len(indices), 0))

    end = time.perf_counter()
    if load_grids:
        print(f"Loaded {len(grids)} grids and attributes into memory.")
    else:
        print(f"Loaded headers and attributes for {len(attrs)} entries.")
    print(f"Elapsed time: {end - start:.2f} s")
    return grids, attrs, headers

def save_training_grids(outpath, sample_grids, sample_attrs, headers):
    with h5py.File(outpath, "w") as fo:
        for key, val in headers.items():
            if key == 'total_saved_grids':
                continue
            fo.create_dataset(key, data=val)
        fo.create_dataset('total_saved_grids', data=len(sample_grids))
        L = sample_grids[0].shape[0]
        nbytes = (L*L + 7)//8
        packed_grids = np.empty((len(sample_grids), nbytes), dtype=np.uint8)
        for i, g in enumerate(sample_grids):
            bits = (g.flatten() > 0).astype(np.uint8)
            packed_grids[i] = np.packbits(bits, bitorder='little')
        fo.create_dataset('grids', data=packed_grids)
        fo.create_dataset('attrs', data=sample_attrs)