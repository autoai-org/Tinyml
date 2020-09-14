import math
from .base import Layer
from .convolution import im2col_indices, get_im2col_indices
from tinynet.core import Backend as np


def col2im_no_dup(cols, x_shape, field_height=3, field_width=3, padding=1,
                  stride=1):
    '''
    Similar function for col2im_indices, but will not perform +=.
    This function is used for 
    '''
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    x_padded[:, k, i, j] = cols_reshaped
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class MaxUnpool2D(Layer):
    def __init__(self, name, input_dim, size, stride):
        super().__init__(name)
        self.type = 'MaxUnpool2D'
        self.input_channel, self.input_height, self.input_width = input_dim
        self.size = size
        self.stride = stride
        self.out_height = (self.input_height - 1) * stride + size[0]
        self.out_width = (self.input_width - 1) * stride + size[1]
        # it is definitely integer, so we do not need to check it anymore
        self.out_dim = (self.input_channel, self.out_height, self.out_width)

    def forward(self, input, max_indices):
        self.num_of_entries = input.shape[0]
        output_shape = (self.num_of_entries,
                        self.out_dim[0], self.out_dim[1], self.out_dim[2])
        indices = max_indices.reshape(input.shape)
        unpooled = np.zeros(output_shape)
        for i in range(self.num_of_entries):
            for j in range(self.input_channel):
                for m in range(self.input_height):
                    for n in range(self.input_width):
                        index = indices[i, j, m, n]
                        w_index = index %  self.size[0]
                        h_index = index // self.size[1]
                        unpooled[i, j,  m*self.stride+h_index, n*self.stride +
                                 w_index] = input[i, j, m, n]
        return unpooled

    def backward(self, in_gradient):
        '''
        This function is not needed in computation, at least right now.
        '''
        pass
    
    def __call__(self, input, max_indices):
        return self.forward(input, max_indices)