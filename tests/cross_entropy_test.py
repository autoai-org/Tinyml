import unittest

import numpy as np

import torch
from tests.base import EPSILON, GRAD_EPSILON
from tinynet.losses import cross_entropy_with_softmax_loss
from torch.nn import CrossEntropyLoss


class TestCrossEntropy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.random.randn(3, 5)
        self.target = np.random.randint(0, 4, (3, ))
        self.torch_ce_loss = CrossEntropyLoss()

    def test(self):
        torch_input = torch.from_numpy(self.data)
        torch_input.requires_grad = True
        torch_output = self.torch_ce_loss(torch_input,
                                          torch.from_numpy(self.target).long())
        tnn_output, tnn_loss_gradient = cross_entropy_with_softmax_loss(
            self.data, self.target)
        torch_output.backward()
        torch_gradient = torch_input.grad.numpy()
        self.assertTrue(
            (torch_output.detach().numpy() - tnn_output < EPSILON).all())
        self.assertTrue(
            (torch_gradient - tnn_loss_gradient < GRAD_EPSILON).all())


if __name__ == '__main__':
    unittest.main()
