import unittest

import numpy as np

import torch
from tests.base import EPSILON
from tinynet.losses import cross_entropy_with_softmax_loss
from torch.nn import CrossEntropyLoss


class TestCrossEntropy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch_size = 1
        classes = 5
        self.data = np.random.randn(1, classes)
        self.target = np.random.randint(0, classes - 1, (batch_size, ))
        self.torch_ce_loss = CrossEntropyLoss()

    def test_forward(self):
        self.torch_input = torch.from_numpy(self.data)
        self.torch_input.requires_grad = True

        self.torch_output = self.torch_ce_loss(
            self.torch_input,
            torch.from_numpy(self.target).long())

        tnn_output, self.tnn_loss_gradient = cross_entropy_with_softmax_loss(
            self.data, self.target)

        self.assertTrue(
            np.absolute(self.torch_output.detach().numpy() -
                        tnn_output < EPSILON).all())

    def test_backward(self):
        self.test_forward()
        self.torch_output.backward()
        torch_gradient = self.torch_input.grad.numpy()

        self.assertTrue((np.absolute(torch_gradient - self.tnn_loss_gradient) <
                         EPSILON).all())


if __name__ == '__main__':
    unittest.main()
