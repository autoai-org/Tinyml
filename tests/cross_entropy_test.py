import unittest
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from tinynet.losses import cross_entropy_with_softmax_loss
from tests.base import EPSILON

class TestCrossEntropy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.random.randn(3,5)
        self.target = np.random.randint(0,4,(3,))
        self.torch_ce_loss = CrossEntropyLoss()
        
    
    def test(self):
        torch_input = torch.from_numpy(self.data)
        torch_output = self.torch_ce_loss(torch_input, torch.from_numpy(self.target).long())
        tnn_output = cross_entropy_with_softmax_loss(self.data, self.target)
        self.assertTrue((torch_input.numpy() - tnn_output < EPSILON).all())

if __name__ == '__main__':
    unittest.main()