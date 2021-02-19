from .base import Layer
from tinynet.core import Backend as np
from .convolution import im2col_indices, col2im_indices

class MaxPool2D(Layer):
    '''
    Perform Max pooling, i.e. select the max item in a sliding window.
    '''
    def __init__(self, name, input_dim, size, stride, return_index=False):
        super().__init__(name)
        self.type = 'MaxPool2D'
        self.input_channel, self.input_height, self.input_width = input_dim
        self.size = size
        self.stride = stride
        self.return_index = return_index
        self.out_height = (self.input_height - size[0]) / stride + 1
        self.out_width = (self.input_width - size[1]) / stride + 1
        if not self.out_height.is_integer() or not self.out_width.is_integer():
            raise Exception("[Tinynet] Invalid dimension settings!")
        self.out_width = int(self.out_width)
        self.out_height = int(self.out_height)
        self.out_dim = (self.input_channel, self.out_height, self.out_width)

    def forward(self, input):
        self.num_of_entries = input.shape[0]
        input_reshaped = input.reshape(input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3])
        self.input_col = im2col_indices(input_reshaped, self.size[0], self.size[1], padding=0, stride=self.stride)
        self.max_indices = np.argmax(self.input_col, axis=0)
        self.total_count = list(range(0, self.max_indices.size))
        output = self.input_col[self.max_indices, self.total_count]
        output = output.reshape(self.out_height, self.out_width, self.num_of_entries, self.input_channel).transpose(2,3,0,1)
        indices = self.max_indices.reshape(self.out_height, self.out_width, self.num_of_entries, self.input_channel).transpose(2,3,0,1)
        if self.return_index:
            return output, indices
        else:
            return output

    def backward(self, in_gradient):
        gradient_col = np.zeros_like(self.input_col)
        gradient_flat = in_gradient.transpose(2,3,0,1).ravel()
        gradient_col[self.max_indices, self.total_count] = gradient_flat
        shape = (self.num_of_entries*self.input_channel, 1, self.input_height, self.input_width)
        out_gradient = col2im_indices(gradient_col, shape, self.size[0], self.size[1], padding=0, stride=self.stride).reshape(self.num_of_entries, self.input_channel, self.input_height, self.input_width)
        return out_gradient