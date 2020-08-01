import unittest

import numpy as np

import torch
from tests.base import EPSILON
from tinynet.layers import Linear as tnn_linear
from torch.nn import Linear as torch_linear


class TestLinearLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch_size = 3
        self.data = np.random.randn(batch_size, 5)
        self.forward_weight = np.random.randn(5, 5)
        self.forward_bias = np.random.randn(batch_size, 5)
        self.torch_linear = torch_linear(5, 5, False)
        self.tnn_linear = tnn_linear('test', 5, 5)
        self.gradient = np.random.randn(batch_size, 5)

    def test_forward(self):
        self.torch_input = torch.from_numpy(self.data)
        self.torch_input.requires_grad = True
        self.torch_linear.weight = torch.nn.Parameter(
            torch.from_numpy(self.forward_weight.T))
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
        self.torch_output.backward(torch.from_numpy(self.gradient))
        output_grad = self.tnn_linear.backward(self.gradient)

        self.assertTrue(
            np.absolute((self.torch_linear.weight.grad.numpy().T -
                         self.tnn_linear.weight.gradient) < EPSILON).all())

        print(self.torch_linear.bias.grad.numpy())
        print(self.tnn_linear.bias.gradient)
        
        self.assertTrue(
            np.absolute((self.torch_linear.bias.grad.numpy() -
                         self.tnn_linear.bias.gradient) < EPSILON).all())
                         
        self.assertTrue(
            np.absolute(
                (self.torch_input.grad.numpy() - output_grad) < EPSILON).all())


if __name__ == '__main__':
    unittest.main()
