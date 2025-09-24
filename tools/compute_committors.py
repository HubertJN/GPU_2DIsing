import numpy as np
import h5py
import gasp
from tools.utils import load_into_array, magnetisation, load_config

config = load_config("config.yaml")

grids, attrs, headers = load_into_array(config.paths.training)

#target = 80.0
#idx = np.argmin(np.abs(attrs[:, 1] - target))
#grids = np.expand_dims(grids[idx], axis=0)
#attrs = np.expand_dims(attrs[idx], axis=0)

#print(attrs)

L = headers['L']
nsweeps = headers['tot_nsweeps']
beta = headers['beta']
h = headers['h']

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult
ngrids = 4 * gpu_nsms * 32
conc_calc = min(int(np.round(ngrids/config.comm.ngrids)), gpu_nsms)

print(f"Running {conc_calc} concurrent calculation on {ngrids} grids")

dn_threshold = config.collective_variable.dn_threshold
up_threshold = config.collective_variable.up_threshold

for start in range(0, len(grids), conc_calc):
    end = min(start + conc_calc, len(grids))
    gridlist = [g.copy() for g in grids[start:end]]
    attrlist = attrs[start:end].copy()

    if attrlist[:, 1].min() < 20:
        mag_output_int = 1
    elif attrlist[:, 1].min() < 40:
        mag_output_int = 10
    else:
        mag_output_int = 100

    pBfast = np.array(
        gasp.run_committor_calc(
            L, ngrids, config.comm.nsweeps, beta, h,
            grid_output_int=50000,
            mag_output_int=mag_output_int,
            grid_input="NumPy",
            grid_array=gridlist,
            cv=config.collective_variable.type,
            dn_threshold=dn_threshold,
            up_threshold=up_threshold,
            store_grids=False,
            nsms=gpu_nsms
        )
    )

    attrlist[:, 2] = pBfast[:, 0]
    attrlist[:, 3] = pBfast[:, 1]

    with h5py.File(config.paths.training, "r+") as fo:
        fo['attrs'][start:end, :] = attrlist
