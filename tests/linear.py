import unittest

import numpy as np

import torch
from tests.base import EPSILON, GRAD_EPSILON
from tinynet.layers import Linear as tnn_linear
from torch.nn import Linear as torch_linear


class TestLinearLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.random.randn(1, 5)
        self.forward_weight = np.random.randn(5, 5)
        self.bias = np.random.randn(1, 5)
        self.torch_linear = torch_linear(5, 5, False)
        self.tnn_linear = tnn_linear('test', 5, 5)
        self.gradient = np.random.randn(1, 5)

    def test_forward(self):
        torch_input = torch.from_numpy(self.data)
        self.torch_linear.weight = torch.nn.Parameter(
            torch.from_numpy(self.forward_weight.T))
        self.tnn_linear.weight.tensor = self.forward_weight
        self.torch_output = self.torch_linear(torch_input)
        self.tnn_output = self.tnn_linear(self.data)
        self.assertTrue((self.torch_output.detach().numpy() - self.tnn_output <
                         EPSILON).all())

    def test_backward(self):
        self.test_forward()
        self.torch_output.backward(torch.from_numpy(self.gradient))
        self.tnn_linear.backward(self.gradient)
        self.assertTrue((self.torch_linear.weight.grad.numpy() -
                         self.tnn_linear.weight.gradient < GRAD_EPSILON).all())


if __name__ == '__main__':
    unittest.main()
