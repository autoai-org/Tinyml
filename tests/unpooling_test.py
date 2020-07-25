import unittest
import numpy as np
from tinynet.layers import MaxPool2D
from tinynet.layers import MaxUnpool2D

class TestMaxpool2D(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_channel_layer = MaxPool2D('test_single', (1, 4, 4), (2,2), 1, return_index=True)
        self.multi_channel_layer = MaxPool2D('test_multi', (3,4,4), (2,2), 1, return_index=False)
        self.single_unpool_layer = MaxUnpool2D('test_unpool', (1,3,3), (2,2), 1)
    
    def test_single_channel_forward(self):
        data = np.array([[1,2,3,4],[5,6,7,8], [1,3,5,7], [2,4,6,8]])
        data = data.reshape(1, 1, data.shape[0], data.shape[1])
        output, indices = self.single_channel_layer(data)
        ground_truth = np.array([[[[6,7,8],[6,7,8],[4,6,8]]]])
        self.assertTrue((output==ground_truth).all())
        input = self.single_unpool_layer(output, indices)
        input_ground_truth = ([[[[0,0,0,0],[0, 6,7,8],[0,0,0,0], [0,4,6,8]]]])
        self.assertTrue((input==input_ground_truth).all())

if __name__ == '__main__':
    unittest.main()