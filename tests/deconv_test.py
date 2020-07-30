import unittest

import numpy as np

import torch
from tinynet.layers import Deconv2D
from torch.nn import ConvTranspose2d as torch_deconv
from tests.base import EPSILON


class TestDeconv2D(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_weight = np.array(
            [[1, 2], [3, 4]], dtype=np.float32).T.reshape(1, 1, 2, 2)
        self.data = np.array([[37, 47], [67, 77]],
                             dtype=np.float32).reshape(1, 1, 2, 2)
        self.torch_deconv = torch_deconv(1, 1, (2, 2), 1, bias=False)
        self.tnn_deconv = Deconv2D('test_deconv', (1, 2, 2), 1, 2, 2, 1, 1)
        self.tnn_deconv.weight.tensor = self.forward_weight
        self.torch_deconv.weight = torch.nn.Parameter(
            torch.from_numpy(self.forward_weight))

    def test_forward(self):
        self.torch_deconv_output = self.torch_deconv(
            torch.from_numpy(self.data))
        tnn_deconv_output = self.tnn_deconv(self.data)
        self.assertTrue(
            (self.torch_deconv_output.detach().numpy() == tnn_deconv_output).all())


class TestDeconv2D_multi_channel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_weight = np.random.randn(6, 3, 18, 18)
        self.forward_bias = np.random.randn(3, )
        self.data = np.random.randn(2, 6, 18, 18)

        self.torch_deconv = torch_deconv(6, 3, (2,2), 1, 0,dilation=1)

        self.tnn_deconv = Deconv2D('test', (6, 18, 18), 3, 2, 2, 1, 1)
        
        self.tnn_deconv.weight.tensor = self.forward_weight
        self.tnn_deconv.bias.tensor = self.forward_bias

        self.torch_deconv.weight = torch.nn.Parameter(
            torch.from_numpy(self.forward_weight))
        self.torch_deconv.bias = torch.nn.Parameter(
            torch.from_numpy(self.forward_bias.flatten()))

    def test_forward(self):
        self.torch_deconv_output = self.torch_deconv(
            torch.from_numpy(self.data))
        tnn_deconv_output = self.tnn_deconv(self.data)
        self.assertTrue(
            (self.torch_deconv_output.detach().numpy() - tnn_deconv_output <
             EPSILON).all())


if __name__ == '__main__':
    unittest.main()
