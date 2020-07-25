import unittest

import numpy as np

import torch
from tinynet.layers import Conv2D, Deconv2D
from torch.nn import Conv2d as torch_conv
from torch.nn import ConvTranspose2d as torch_deconv


class TestDeconv2D(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_weight = np.array([[1, 2], [3, 4]],
                                       dtype=np.float32).reshape(1, 1, 2, 2)
        self.forward_bias = np.random.rand(12, 1)
        self.data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                             dtype=np.float32).reshape(1, 1, 3, 3)
        self.tnn_conv = Conv2D('test_conv2d', (1, 3, 3), 1, 2, 2, 1, 0)
        self.torch_conv = torch_conv(1, 1, (2, 2), 1, 0)
        self.tnn_conv.weight.tensor = self.forward_weight
        # self.tnn_conv.bias.tensor = self.forward_bias

    def test_convolution(self):
        torch_conv_output = None
        with torch.no_grad():
            self.torch_conv.weight = torch.nn.Parameter(
                torch.from_numpy(self.forward_weight))
            # self.torch_conv.bias = torch.nn.Parameter(torch.from_numpy(self.forward_bias))
            torch_conv_output = self.torch_conv(torch.from_numpy(self.data))
        tnn_conv_output = self.tnn_conv(self.data)
        print(tnn_conv_output.shape)
        print(torch_conv_output.shape)
        print(tnn_conv_output)
        print(torch_conv_output)
        self.assertTrue((torch_conv_output.numpy() == tnn_conv_output).all())


class TestDeconv2D_multiple(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_weight = np.random.rand(12, 3, 2, 2)
        self.forward_bias = np.random.rand(12, 1)
        self.data = np.random.rand(1, 3, 8, 8)
        self.tnn_conv = Conv2D('test_conv2d', (3, 8, 8), 12, 2, 2, 1, 0)
        self.torch_conv = torch_conv(3, 12, (2, 2), 1, 0)
        self.tnn_conv.weight.tensor = self.forward_weight
        # self.tnn_conv.bias.tensor = self.forward_bias

    def test_convolution(self):
        tnn_conv_output = self.tnn_conv(self.data)
        torch_conv_output = None
        with torch.no_grad():
            self.torch_conv.weight = torch.nn.Parameter(
                torch.from_numpy(self.forward_weight))
            # self.torch_conv.bias = torch.nn.Parameter(torch.from_numpy(self.forward_bias))
            torch_conv_output = self.torch_conv(torch.from_numpy(self.data))
        print(tnn_conv_output.shape)
        print(torch_conv_output.shape)
        print(tnn_conv_output)
        print(torch_conv_output)
        self.assertTrue((torch_conv_output.numpy() == tnn_conv_output).all())


if __name__ == '__main__':
    unittest.main()
