import unittest

import numpy as np
import torch
from torch.nn import ReLU as torch_relu

from tinyml.layers import ReLu


class TestRelu(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tnn_relu = ReLu('relu')
        self.torch_relu = torch_relu()
        self.data = np.random.rand(3, 6, 8, 9)
        self.gradient = np.random.rand(3, 6, 8, 9)

    def test_forward(self):
        self.torch_input = torch.from_numpy(self.data)
        self.torch_input.requires_grad = True

        self.torch_output = self.torch_relu(self.torch_input)
        tnn_output = self.tnn_relu(self.data)
        self.assertTrue(
            (self.torch_output.detach().numpy() == tnn_output).all())

    def test_backward(self):
        self.test_forward()
        self.torch_output.backward(torch.from_numpy(self.gradient))

        out_grad = self.tnn_relu.backward(self.gradient)

        self.assertTrue(
            (self.torch_input.grad.detach().numpy() == out_grad).all())


if __name__ == '__main__':
    unittest.main()
