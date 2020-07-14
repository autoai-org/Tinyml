from .base import Layer
from tinynet.core import Backend as np
from .convolution import im2col_indices, col2im_indices

class MaxPool2D(Layer):
    '''
    Perform Max pooling.
    '''
    def __init__(self, name, input_dim, size, stride, return_indx=False):
        super().__init__(name)
        self.input_channel, self.input_height, self.input_width = input_dim
        self.size = size
        self.stride = stride
        self.return_index = self.return_index
        self.out_height = (self.input_height - size) / stride + 1
        self.out_width = (self.input_width - size) / stride + 1
        if not self.out_height.is_integer() or not self.out_width.is_integer():
            raise Exception("[Tinynet] Invalid dimension settings!")
        self.out_width = int(self.out_width)
        self.out_height = int(self.out_height)
        
    def forward(self, input):
        pass
    def backward(self, in_gradient):
        pass