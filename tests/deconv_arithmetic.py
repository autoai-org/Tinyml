import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.nn import Conv2d, ConvTranspose2d

# Convolution
torch_conv = Conv2d(1, 1, (2, 2), 2, bias=False)
filter = torch.from_numpy(np.arange(1, 5, dtype='float32').reshape(1, 1, 2, 2))
filter_matrix = torch.from_numpy(
    np.array([
        [
            1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ],
        [
            0, 0, 1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0,
            0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 0, 0, 0,
            0, 0, 0
        ],
    ],
             dtype='float32').reshape(1, 1, 4, 25))

torch_conv.weight = torch.nn.Parameter(filter)

input = torch.from_numpy(np.arange(1, 26, dtype='float32').reshape(1, 1, 5, 5))

conv_output = torch_conv(input)
print('convolution output is:')
print(conv_output)
print('matrix multiplication output')
matmul_output = torch.matmul(filter_matrix,
                             input.reshape(1, 1, 25, 1)).reshape(1, 1, 2, 2)
print(matmul_output)
assert torch.all(conv_output.eq(matmul_output))

# Deconvolution
torch_deconv = ConvTranspose2d(1, 1, (2, 2), 2, bias=False)
deconv_filter = filter
deconv_filter_matrix = filter_matrix.transpose(2, 3)

torch_deconv.weight = torch.nn.Parameter(deconv_filter)

deconv_output = torch_deconv(conv_output, output_size=input.shape)
print('deconvolution output is:')
print(deconv_output)

deconv_matmul_output = torch.matmul(deconv_filter_matrix,
                                    conv_output.reshape(1, 1, 4, 1)).reshape(
                                        1, 1, 5, 5)

print(deconv_matmul_output)
assert torch.all(deconv_output.eq(deconv_matmul_output))
