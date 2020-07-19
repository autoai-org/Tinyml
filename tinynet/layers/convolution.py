from .base import Layer
from tinynet.core import Backend as np
from tinynet.core import GPU

'''
These im2col and col2im should be credited to:
https://github.com/huyouare/CS231n. Some minor modifications are made to ensure
it works on GPU.
> TODO: add a naive implementation.
'''

def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(int(out_height), dtype='int32'), int(out_width))
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'),
                  field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    if GPU:
        # In cupy, scatter_add is equivalent to np.add.at
        np.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    else:
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D(Layer):
    '''
    Conv2D performs convolutional operation with given input.
    '''
    def __init__(self, name, input_dim, n_filter, h_filter, w_filter, stride, padding):
        '''
        input_dim:

        n_filter:

        h_filter:

        w_filter:

        stride: 
        '''
        super().__init__(name)
        self.type = 'Conv2D'
        self.input_channel, self.input_height, self.input_weight = input_dim
        self.n_filter = n_filter
        self.h_filter = h_filter
        self.w_filter = w_filter
        self.stride = stride
        self.padding = padding
        weight = np.random.randn(
            self.n_filter, self.input_channel, self.h_filter, self.w_filter) / np.sqrt(n_filter/2.0)
        bias = np.zeros((self.n_filter, 1))
        self.weight = self.build_param(weight)
        self.bias = self.build_param(bias)
        self.out_height = (self.input_height - self.h_filter +
                      2 * padding) / self.stride + 1
        self.out_width = (self.input_weight - self.w_filter +
                      2 * padding) / self.stride + 1
        if not self.out_width.is_integer() or not self.out_height.is_integer():
            raise Exception("[Tinynet] Invalid dimension settings!")
        self.out_height, self.out_width = int(self.out_height), int(self.out_width)
        self.out_dim = (self.n_filter, self.out_height, self.out_width)

    def forward(self, input):
        self.n_input = input.shape[0]
        self.input_col = im2col_indices(
            input, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        weight_in_row = self.weight.tensor.reshape(self.n_filter, -1)
        output = np.matmul(weight_in_row, self.input_col) + self.bias.tensor
        output = output.reshape(
            self.n_filter, self.out_height, self.out_width, self.n_input)
        output = output.transpose(3, 0, 1, 2)
        return output

    def backward(self, in_gradient):
        gradient_flat = in_gradient.transpose(
            1, 2, 3, 0).reshape(self.n_filter, -1)
        weight_gradient = np.matmul(gradient_flat, self.input_col.T)
        self.weight.gradient = weight_gradient.reshape(
            self.weight.tensor.shape)
        self.bias.gradient = np.sum(in_gradient, axis=(
            0, 2, 3)).reshape(self.n_filter, -1)
        weight_flat = self.weight.tensor.reshape(self.n_filter, -1)
        out_gradient_col = np.matmul(weight_flat.T, gradient_flat)
        shape = (self.n_input, self.input_channel,
                 self.input_height, self.input_weight)
        out_gradient = col2im_indices(out_gradient_col, shape, self.h_filter,
                                      self.w_filter, self.padding, self.stride)
        return out_gradient
