import os

GPU = False

if "TNN_GPU" in os.environ:
    GPU=True

Backend = None

if GPU:
    print('[Tinynet] Using GPU as backend')
    import cupy as cp
    Backend = cp
else:
    import numpy as np
    Backend = np