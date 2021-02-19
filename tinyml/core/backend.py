import os

GPU = False

if "TNN_GPU" in os.environ:
    GPU=True

Backend = None

if GPU:
    print('[tinyml] Using GPU as backend')
    try:
        import cupy as cp
        Backend = cp
    except Exception as e:
        print('no cupy found, will fallback to cpu mode')
        import numpy as np
        Backend = np
else:
    import numpy as np
    Backend = np