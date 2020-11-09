import sys
sys.path.append('.')
sys.path.append('build')
import gunrock_sssp

import numpy as np
from scipy.io import mmread

# Load graph
csr = mmread('chesapeake.mtx').tocsr()

# Allocate host memory for output
distances    = np.zeros(csr.shape[0]).astype(np.float32)
predecessors = np.zeros(csr.shape[0]).astype(np.int32)

# Run
gunrock_sssp.gunrock_sssp(csr.shape[0], csr.nnz, csr.indptr, csr.indices, csr.data, 0, distances, predecessors)

# Print output
print('Distances:', distances)
print('Predecessors:', predecessors)