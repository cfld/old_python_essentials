import torch

import sys
sys.path.append('.')
sys.path.append('build')
from example_app import *

x = torch.randn(2, 2)
print(x)

example_app(x)