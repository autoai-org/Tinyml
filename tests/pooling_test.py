import unittest

import numpy as np

import torch
from tinyml.layers import MaxPool2D
from torch.nn import MaxPool2d as torch_max_pool_2d


class TestMaxpool2D(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_channel_layer = MaxPool2D('test_single', (1, 4, 4), (2, 2),
                                              1,
                                              return_index=False)
        self.multi_channel_layer = MaxPool2D('test_multi', (3, 4, 4), (2, 2),
                                             1,
                                             return_index=False)

    def test_single_channel_forward(self):
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7],
                         [2, 4, 6, 8]])
        data = data.reshape(1, 1, data.shape[0], data.shape[1])
        output = self.single_channel_layer(data)
        ground_truth = np.array([[[[6, 7, 8], [6, 7, 8], [4, 6, 8]]]])
        self.assertTrue((output == ground_truth).all())


class TestMaxpool2DPyTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.random.randn(2, 3, 6, 6)
        self.torch_layer = torch_max_pool_2d((2, 2),
                                             1,
                                             padding=0,
                                             dilation=1,
                                             return_indices=False)
        self.tnn_layer = MaxPool2D('test', (3, 6, 6), (2, 2),
                                   1,
                                   return_index=False)

    def test(self):
        torch_output = self.torch_layer(torch.from_numpy(self.data))
        tnn_output = self.tnn_layer(self.data)
        self.assertTrue((torch_output.numpy() == tnn_output).all())


if __name__ == '__main__':
    unittest.main()
