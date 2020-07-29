import unittest

import numpy as np

import torch
from tests.base import EPSILON, GRAD_EPSILON
from tinynet.losses import mse_loss
from torch.nn import MSELoss


class TestMSE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.random.randn(5, 5)
        self.output = np.random.randn(5, 5)
        self.torch_mse_loss = MSELoss()

    def test(self):
        torch_input = torch.from_numpy(self.data)
        torch_input.requires_grad = True
        torch_output = self.torch_mse_loss(
            torch_input,
            torch.from_numpy(self.output).long())
        print(torch_output)
        tnn_output, tnn_loss_gradient = mse_loss(self.data, self.output)
        print(tnn_output)


if __name__ == '__main__':
    unittest.main()
