import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

file = open('bin/index.bin', 'rb')
#file = open('bin/committor_index.bin', 'rb')

while True:
    first = np.fromfile(file, dtype=np.int32, count=3, sep='')
    last = np.fromfile(file, dtype=np.float64, count=2)
    
    print(first, last)

    # check
    if first.size < 3:
        break
