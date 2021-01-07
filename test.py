#!/usr/bin/env python

"""
  test.py
"""

import torch

import sys
sys.path.append('.')
sys.path.append('build')
import pygunrock as pyg

import numpy as np
from time import time
from tqdm import trange
from scipy.io import mmread

np.set_printoptions(linewidth=240)

# Load graph
csr = mmread('chesapeake.mtx').tocsr()
# csr = mmread('cit-Patents-sub.mtx').tocsr()

n_vertices = csr.shape[0]
n_edges    = csr.nnz

# Convert data to torch + move to GPU
indptr   = torch.IntTensor(csr.indptr).cuda()
indices  = torch.IntTensor(csr.indices).cuda()
data     = torch.FloatTensor(csr.data).cuda()

# Allocate memory for output
distances    = torch.zeros(csr.shape[0]).float().cuda()
predecessors = torch.zeros(csr.shape[0]).int().cuda()

# Create graph
for single_source in [0, 1, 2]:
  distances.zero_()
  predecessors.zero_()

  G = pyg.from_csr(n_vertices, n_vertices, n_edges, indptr, indices, data)
  _ = pyg.sssp(G, single_source, distances, predecessors)
  print(distances)