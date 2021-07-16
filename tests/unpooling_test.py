import unittest

import numpy as np
import torch
from torch.nn import MaxPool2d as torch_max_pool_2d
from torch.nn import MaxUnpool2d as torch_max_unpool_2d

from tests.base import EPSILON
from tinyml.layers import MaxPool2D, MaxUnpool2D


class TestMaxUnpool2D(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_channel_layer = MaxPool2D('test_single', (1, 4, 4), (2, 2),
                                              1,
                                              return_index=True)
        self.multi_channel_layer = MaxPool2D('test_multi', (3, 4, 4), (2, 2),
                                             1,
                                             return_index=False)
        self.single_unpool_layer = MaxUnpool2D('test_unpool', (1, 3, 3),
                                               (2, 2), 1)

    def test_single_channel_forward(self):
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7],
                         [2, 4, 6, 8]])
        data = data.reshape(1, 1, data.shape[0], data.shape[1])
        output, indices = self.single_channel_layer(data)
        ground_truth = np.array([[[[6, 7, 8], [6, 7, 8], [4, 6, 8]]]])
        self.assertTrue((output == ground_truth).all())
        unpooled = self.single_unpool_layer(output, indices)
        input_ground_truth = ([[[[0, 0, 0, 0], [0, 6, 7, 8], [0, 0, 0, 0],
                                 [0, 4, 6, 8]]]])
        self.assertTrue((unpooled == input_ground_truth).all())


class TestUnpool2DwithTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.random.rand(2, 3, 20, 20)
        self.torch_max_pool_layer = torch_max_pool_2d((2, 2), 2, 0, 1, True)
        self.tnn_pool_layer = MaxPool2D('prepare', (3, 20, 20), (2, 2),
                                        2,
                                        return_index=True)
        self.torch_layer = torch_max_unpool_2d((2, 2), 2, padding=0)
        self.tnn_layer = MaxUnpool2D('test', (3, 10, 10), (2, 2), 2)

    def test(self):
        torch_output, torch_indices = self.torch_max_pool_layer(
            torch.from_numpy(self.data))
        tnn_output, tnn_indices = self.tnn_pool_layer(self.data)
        self.assertTrue((torch_output.numpy() == tnn_output).all())
        tnn_input = self.tnn_layer(tnn_output, tnn_indices)
        torch_input = self.torch_layer(torch_output, torch_indices)
        self.assertTrue((torch_input.numpy() == tnn_input).all())


class TestUnpool2DwithStride(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_channel_layer = MaxPool2D('test_single', (1, 4, 4), (2, 2),
                                              2,
                                              return_index=True)
        self.single_unpool_layer = MaxUnpool2D('test_unpool', (1, 2, 2),
                                               (2, 2), 2)

    def test_single_channel_forward(self):
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7],
                         [2, 4, 6, 8]])
        data = data.reshape(1, 1, data.shape[0], data.shape[1])
        output, indices = self.single_channel_layer(data)
        ground_truth = np.array([[[[6, 8], [4, 8]]]])
        self.assertTrue((output == ground_truth).all())
        unpooled = self.single_unpool_layer(output, indices)
        input_ground_truth = ([[[[0, 0, 0, 0], [0, 6, 0, 8], [0, 0, 0, 0],
                                 [0, 4, 0, 8]]]])
        self.assertTrue((unpooled == input_ground_truth).all())


if __name__ == '__main__':
    unittest.main()
