import unittest

import numpy as np

import torch
from tests.base import EPSILON
from tinyml.layers import Linear as tnn_linear
from torch.nn import Linear as torch_linear


class TestLinearLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch_size = 3
        in_features = 5
        out_features = 2
        self.data = np.random.randn(batch_size, in_features)
        self.forward_weight = np.random.randn(out_features, in_features)
        self.forward_bias = np.random.randn(out_features)
        self.torch_linear = torch_linear(in_features, out_features)
        self.tnn_linear = tnn_linear('test', in_features, out_features)

        self.gradient = np.random.randn(batch_size, out_features)

        self.torch_input = torch.from_numpy(self.data)
        self.torch_input = self.torch_input.view(self.torch_input.size(0), -1)
        self.torch_input.requires_grad = True
        self.torch_input.retain_grad()

    def test_forward(self):

        self.torch_linear.weight = torch.nn.Parameter(
            torch.from_numpy(self.forward_weight))
        self.torch_linear.bias = torch.nn.Parameter(
            torch.from_numpy(self.forward_bias))

        self.tnn_linear.weight.tensor = self.forward_weight
        self.tnn_linear.bias.tensor = self.forward_bias

        self.torch_output = self.torch_linear(self.torch_input)
        self.tnn_output = self.tnn_linear(self.data)

        self.assertTrue(
            np.absolute(self.torch_output.detach().numpy() -
                        self.tnn_output < EPSILON).all())

    def test_backward(self):
        self.test_forward()

        output_grad = self.tnn_linear.backward(self.gradient)

        self.torch_output.backward(torch.from_numpy(self.gradient))

        self.assertTrue(
            np.absolute((self.torch_input.grad - output_grad) < EPSILON).all())

        self.assertTrue(
            np.absolute((self.torch_linear.weight.grad.numpy() -
                         self.tnn_linear.weight.gradient) < EPSILON).all())

        self.assertTrue(
            np.absolute((self.torch_linear.bias.grad.numpy() -
                         self.tnn_linear.bias.gradient) < EPSILON).all())


if __name__ == '__main__':
    unittest.main()
