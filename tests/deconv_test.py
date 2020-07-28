import unittest

import numpy as np

import torch
from tinynet.layers import Conv2D, Deconv2D
from torch.nn import Conv2d as torch_conv
from torch.nn import ConvTranspose2d as torch_deconv


class TestDeconv2D_multiple(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_weight = np.random.rand(12, 3, 2, 2)
        self.forward_bias = np.random.rand(12, 1)
        self.data = np.random.rand(1, 3, 8, 8)
        self.tnn_conv = Conv2D('test_conv2d', (3, 8, 8), 12, 2, 2, 1, 0)
        self.torch_conv = torch_conv(3, 12, (2, 2), 1, 0)
        self.tnn_conv.weight.tensor = self.forward_weight
        self.tnn_conv.bias.tensor = self.forward_bias
        self.tnn_deconv = Deconv2D('test_deconv', (12, 7, 7), 3, 2, 2, 1, dilation=1)
        self.tnn_deconv.weight.tensor = self.tnn_conv.weight.tensor
        self.tnn_deconv.bias.tensor = self.tnn_conv.bias.tensor

    def test_convolution(self):
        print(self.data.shape)
        tnn_conv_output = self.tnn_conv(self.data)
        tnn_deconv_output = self.tnn_deconv(tnn_conv_output)
        print(tnn_conv_output.shape)
        print(tnn_deconv_output.shape)


if __name__ == '__main__':
    unittest.main()
