import os

GPU = os.environ['TNN_GPU']

Backend = None

if GPU:
    print('[Tinynet] Using GPU as backend')
    import cupy as cp
    Backend = cp
else:
    import numpy as np
    Backend = np