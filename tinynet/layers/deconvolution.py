from .base import Layer
from .convolution import im2col_indices, Conv2D
from tinynet.core import Backend as np

class Deconv2D(Layer):
    '''
    Deconv2D performs deconvolutional operation, or transposed convolution.
    '''
    def __init__(self, name, input_dim, n_filter, h_filter, w_filter, stride, padding):
        '''
        :param input_dim: the input data dimension, it should be of the shape (C, H, W) where C is for the channel, H for height and W for width.

        :param n_filter: the number of filters used in this layer. It can be any integers.

        :param h_filter: the height of the filter.

        :param w_filter: the width of the filter.

        :param stride: the stride of the sliding filter.

        The output shape will be:
        .. math::

        (stride * (input_size - 1) + dilation * (h_filter - 1) + 1, 
        stride * (input_size - 1) + dilation * (w_filter - 1) + 1)

        '''
        super().__init__(name)
        self.type = 'Deconv2D'
        self.input_channel, self.input_height, self.input_width = input_dim
        self.n_filter = n_filter
        self.h_filter = h_filter
        self.w_filter = w_filter
        self.stride = stride
        self.padding = padding
        weight = np.random.randn(
            self.n_filter, self.input_channel, self.h_filter, self.w_filter) / np.sqrt(n_filter/2.0)
        bias = np.zeros((self.n_filter,1))
        self.weight = self.build_param(weight)
        self.bias = self.build_param(bias)
        self.out_height = max(0, self.input_height.shape[2] - self.h_filter + 1)
        self.out_width = max(0, self.input_width.shape[2] - self.w_filter + 1)
        if not self.out_width.is_integer() or not self.out_height.is_integer():
            raise Exception("[Tinynet] Invalid dimension settings!")
        self.out_height, self.out_width = int(
            self.out_height), int(self.out_width)
        self.out_dim = (self.n_filter, self.out_height, self.out_width)
        self._conv2d = Conv2D('inner_conv2d', input_dim, n_filter, h_filter, w_filter, stride, padding)

    def forward(self, input):
        self.n_input = input.shape[0]
        self.input_col = im2col_indices(
            input, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding
        )
        weight_rotated = np.filp(self.weight.tensor,(2,3))
        weight_rotated_flat = weight_rotated.reshape(self.n_filter, -1)
        output = np.matmul(weight_rotated_flat, self.input_col) + self.bias.tensor
        output = output.T.reshape(self.n_filter, self.out_height, self.out_width, self.n_input)
        output = output.transpose(3,0,1,2)
        return output

    def backward(self, in_gradient):
        '''
        the backward of deconvolution is implemented by a forward pass of convolution.
        '''
        gradient_flat = in_gradient.transpose(1,2,3,0).reshape(self.n_filter, -1)
        weight_gradient = np.matmul(gradient_flat, self.input_col.T).reshape(self.weight.tensor.shape)
        self.weight.gradient = np.flip(weight_gradient, (2,3))
        self.bias.gradient = np.sum(in_gradient, (0,2,3)).reshape(self.n_filter,-1)
        self._conv2d.weight = self.weight
        self._conv2d.bias = self.bias
        return self._conv2d(in_gradient)



