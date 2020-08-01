import unittest

import numpy as np

import torch
from tests.base import EPSILON, GRAD_EPSILON
from tinynet.layers import Conv2D
from torch.nn import Conv2d as torch_conv2d


class TestConv2D(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_weight = np.array([[1, 2], [3, 4]],
                                       dtype=np.float32).reshape(1, 1, 2, 2)
        self.forward_bias = np.random.rand(12, 1)
        self.data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                             dtype=np.float32).reshape(1, 1, 3, 3)
        self.tnn_conv = Conv2D('test_conv2d', (1, 3, 3), 1, 2, 2, 1, 0)
        self.tnn_conv.weight.tensor = self.forward_weight
        self.torch_conv = torch_conv2d(1, 1, (2, 2), 1, bias=False)

    def test_convolution(self):
        self.torch_conv.weight = torch.nn.Parameter(
            torch.from_numpy(self.forward_weight))
        torch_conv_output = self.torch_conv(torch.from_numpy(self.data))
        tnn_conv_output = self.tnn_conv(self.data)
        self.assertTrue(
            (torch_conv_output.detach().numpy() == tnn_conv_output).all())


class TestConv2D_multiple_channel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_weight = np.random.randn(12, 3, 2, 2)
        self.forward_bias = np.random.randn(12, 1)
        self.data = np.random.randn(1, 3, 8, 8)
        self.tnn_conv = Conv2D('test_conv2d', (3, 8, 8), 12, 2, 2, 1, 0)
        self.gradient = np.random.randn(1, 12, 7, 7)
        self.tnn_conv.weight.tensor = self.forward_weight
        self.tnn_conv.bias.tensor = self.forward_bias
        self.torch_conv = torch_conv2d(3, 12, (2, 2), 1, bias=False)

    def test_forward(self):
        tnn_conv_output = self.tnn_conv(self.data)
        self.torch_conv.weight = torch.nn.Parameter(
            torch.from_numpy(self.forward_weight))
        self.torch_conv.bias = torch.nn.Parameter(
            torch.from_numpy(self.forward_bias.flatten()))

        self.torch_conv_output = self.torch_conv(torch.from_numpy(self.data))
        self.assertTrue(
            (self.torch_conv_output.detach().numpy() - tnn_conv_output <
             EPSILON).all())

    def test_backward(self):
        self.test_forward()
        self.torch_conv_output.backward(torch.from_numpy(self.gradient))
        self.tnn_conv.backward(self.gradient)
        self.assertTrue((self.torch_conv.weight.grad.numpy() -
                         self.tnn_conv.weight.gradient < GRAD_EPSILON).all())
        self.assertTrue(
            (self.torch_conv.bias.grad.numpy() - self.tnn_conv.bias.gradient.T
             < GRAD_EPSILON).all())


if __name__ == '__main__':
    unittest.main()
