import torch

import sys
sys.path.append('.')
sys.path.append('build')
from gunrock_sssp import *

import numpy as np
from time import time
from tqdm import trange
from scipy.io import mmread

np.set_printoptions(linewidth=240)

# Load graph
csr = mmread('chesapeake.mtx').tocsr()
# csr = mmread('/home/ubuntu/projects/gunrock/dataset/large/delaunay_n13/delaunay_n13.mtx').tocsr()

n_vertices = csr.shape[0]
n_edges    = csr.nnz

# --

indptr   = torch.IntTensor(csr.indptr).cuda()
indices  = torch.IntTensor(csr.indices).cuda()
data     = torch.FloatTensor(csr.data).cuda()

# Allocate host memory for output
distances    = torch.zeros(csr.shape[0]).float().cuda()
predecessors = torch.zeros(csr.shape[0]).int().cuda()

for single_source in trange(20):
  _ = pt_sssp(n_vertices, n_edges, indptr, indices, data, single_source, distances, predecessors)
  torch.cuda.synchronize()
  print(distances.cpu().numpy()[:10])
