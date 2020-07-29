import unittest

import numpy as np

import torch
from tinynet.layers import ReLu
from torch.nn import ReLU as torch_relu


class TestRelu(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tnn_relu = ReLu('relu')
        self.torch_relu = torch_relu()
        self.data = np.random.rand(2, 3, 4, 5)
        self.gradient = np.random.rand(2,3,4,5)

    def test_forward(self):
        self.torch_output = self.torch_relu(torch.from_numpy(self.data))
        tnn_output = self.tnn_relu(self.data)
        self.assertTrue((self.torch_output.numpy() == tnn_output).all())        

if __name__ == '__main__':
    unittest.main()
