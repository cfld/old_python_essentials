import sys
sys.path.append('.')
sys.path.append('build')
from gunrock_sssp import gunrock_sssp

import numpy as np
from scipy.io import mmread

# weight_type = np.int32
weight_type = np.float32 # Wrapper works w/ either

np.set_printoptions(linewidth=240)

# Load graph
csr = mmread('chesapeake.mtx').tocsr()

n_vertices = csr.shape[0]
n_edges    = csr.nnz

csr.data = csr.data.astype(weight_type)

# Allocate host memory for output
distances    = np.zeros(csr.shape[0]).astype(weight_type)
predecessors = np.zeros(csr.shape[0]).astype(np.int32)

# Run + print output
for single_source in range(csr.shape[0]):
  gunrock_sssp(n_vertices, n_edges, csr.indptr, csr.indices, csr.data, single_source, distances, predecessors)
  print(distances)

# !! This copies data to/from GPU too many times ... how can we work around?
#    Should allow user to create graph object in Python, but having issues w/ types