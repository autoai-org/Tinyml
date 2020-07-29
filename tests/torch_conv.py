import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

forward_weight = np.array([[1, 2], [3, 4]],
                          dtype=np.float32).reshape(1, 1, 2, 2)
forward_bias = np.random.rand(12, 1)
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                dtype=np.float32).reshape(1, 1, 3, 3)

torch_conv = nn.Conv2d(1, 1, (2, 2), 1, 0, padding_mode='zeros', bias=False)

with torch.no_grad():
    torch_conv.weight = torch.nn.Parameter(torch.from_numpy(forward_weight))
    # self.torch_conv.bias = torch.nn.Parameter(torch.from_numpy(self.forward_bias))
    torch_conv_output = torch_conv(torch.from_numpy(data))

print(torch_conv_output)
