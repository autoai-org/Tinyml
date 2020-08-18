from .base import Layer
from tinynet.core import Backend as np


def im2rows(input, inp_shape, filter_shape, dilation, stride, dilated_shape, padding, res_shape):
    """
    Gradient transformation for the im2rows operation
    :param in_gradient: The grad from the next layer
    :param inp_shape: The shape of the input image
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The reformed gradient of the shape of the image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros(inp_shape, dtype=input.dtype)
    input = input.reshape(
        (input.shape[0], input.shape[1], filter_shape[1], filter_shape[2], filter_shape[3]))
    for it in range(num_rows * num_cols):
        # first found index of rows and columns
        # i for rows
        # j for columns
        i = it // num_rows
        j = it % num_rows
        # accessing via colons: [start:end:step]
        # commas are for different dimensions
        res[:, :, i * stride[0]:i * stride[0] + dilated_rows:dilation,
            j * stride[1]:j * stride[1] + dilated_cols:dilation] += input[:, it, :, :, :]
    if (padding != 0):
        # TODO: this only works for pad=1, right now.
        # remove the padding regions
        res = np.delete(res, 0, 2)
        res = np.delete(res, res.shape[2]-1, 2)
        res = np.delete(res, 0, 3)
        res = np.delete(res, res.shape[3]-1,3)
    return res


class Deconv2D(Layer):
    '''
    Deconv2D performs deconvolution operation, or tranposed convolution.    
    '''

    def __init__(self, name, input_dim, n_filters, h_filter, w_filter, stride, dilation=1, padding=0):
        '''
        :param input_dim: the input dimension, in the format of (C,H,W)
        :param n_filters: the number of convolution filters
        :param h_filter: the height of the filter
        :param w_filter: the width of the filter
        :param stride: the stride for forward convolution
        :param dilation: the dilation factor for the filters, =1 by default.
        '''
        super().__init__(name)
        self.type = 'Deconv2D'
        self.input_channel, self.input_height, self.input_width = input_dim
        self.n_filters = n_filters
        self.h_filter = h_filter
        self.w_filter = w_filter
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        weight = np.random.randn(
            self.n_filters, self.input_channel, self.h_filter, self.w_filter) / np.sqrt(self.n_filters/2.0)
        bias = np.zeros((self.n_filters, 1))
        self.weight = self.build_param(weight)
        self.bias = self.build_param(bias)

    def forward(self, input):
        filter_shape = self.weight.tensor.shape
        dilated_shape = (
            (filter_shape[2] - 1) * self.dilation + 1, (filter_shape[3] - 1) * self.dilation + 1)
        res_shape = (
            (self.input_height - 1) * self.stride + dilated_shape[0],
            (self.input_width - 1) * self.stride + dilated_shape[1]
        )
        input_mat = input.reshape(
            (input.shape[0], input.shape[1], -1)).transpose((0, 2, 1))
        filters_mat = self.weight.tensor.reshape(
            self.input_channel, -1)
        res_mat = np.matmul(input_mat, filters_mat)

        return im2rows(res_mat, (input.shape[0], filter_shape[1], res_shape[0], res_shape[1]), filter_shape, self.dilation, (self.stride, self.stride), dilated_shape, self.padding, input.shape[2:])

    def backward(self, in_gradient):
        '''
        This function is not needed in computation, at least right now.
         '''
        return in_gradient
