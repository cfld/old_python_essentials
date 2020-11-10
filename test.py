import torch

import sys
sys.path.append('.')
sys.path.append('build')
from gunrock_sssp import gunrock_sssp

import numpy as np
from time import time
from tqdm import trange
from scipy.io import mmread

np.set_printoptions(linewidth=240)

# Load graph
csr = mmread('chesapeake.mtx').tocsr()

n_vertices = csr.shape[0]
n_edges    = csr.nnz

# --

# indptr   = torch.IntTensor(csr.indptr).cuda()
# indices  = torch.IntTensor(csr.indices).cuda()
# data     = torch.FloatTensor(csr.data).cuda()

# # Allocate host memory for output
# distances    = torch.zeros(csr.shape[0]).float().cuda()
# predecessors = torch.zeros(csr.shape[0]).int().cuda()

# for single_source in range(n_vertices):
#   _ = gunrock_sssp(n_vertices, n_edges, indptr, indices, data, single_source, distances, predecessors)
#   torch.cuda.synchronize()
#   print(distances.cpu().numpy())

# --

import numpy as np
from gunrock_sssp import *

indptr   = torch.IntTensor(csr.indptr)
indices  = torch.IntTensor(csr.indices)
data     = torch.FloatTensor(csr.data)

g = h_graph(n_vertices, n_vertices, n_edges, indptr, indices, data)
g.get_number_of_vertices()
g.get_number_of_edges()

g = d_graph(n_vertices, n_vertices, n_edges, indptr, indices, data)
g.get_number_of_vertices()
g.get_number_of_edges()