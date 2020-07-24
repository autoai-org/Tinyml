from .base import Layer
from tinynet.core import Backend as np
from tinynet.core import GPU
from .convolution import im2col_indices, col2im_indices


class Deconv2D(Layer):
    '''
    Deconvolution (2D) Layer, or transpose convolution.
    The output shape will be:

    .. math::

        (stride * (input_size - 1) + dilation * (h_filter - 1) + 1, 
        stride * (input_size - 1) + dilation * (w_filter - 1) + 1)

    '''

    def __init__(self, name, input_dim, n_filter, h_filter, w_filter, stride, padding=0, dilation=1):
        '''
        :param name
        :param input_dim
        :param n_filter
        :param h_filter
        :param w_filter
        :param stride
        :param padding
        :param dilation [default=1]
        '''
        super().__init__(name)
        self.type = 'Deconv2D'
        self.input_channel, self.input_height, self.input_weight = input_dim
        self.stride = stride
        self.h_filter = h_filter
        self.w_filter = w_filter
        self.n_filter = n_filter
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        weight = np.random.randn(
            self.n_filter, self.input_channel, self.h_filter, self.w_filter) / np.sqrt(n_filter/2.0)
        bias = np.zeros((self.n_filter, 1))
        self.weight = self.build_param(weight)
        self.bias = self.build_param(bias)

    def forward(self, input):
        '''
        input is of shape [N, C, H, W]
        '''
        if self.stride != 1:
            h_stride = input.shape[2] * self.stride
            w_stride = input.shape[3] * self.stride
            input_stride = np.zeros((*input.shape[:2], h_stride, w_stride))
            for height in range(input.shape[2]):
                for width in range(input.shape[3]):
                    input_stride[:, :, height*self.stride, width *
                                 self.stride] = input[:, :, height, width]
            input = input_stride[:, :, :-self.stride+1, :-self.stride+1]
            # check if needs padding in height
            h_padding = self.h_filter - 1 - self.padding
            if h_padding == 0:
                pass
            elif h_padding < 0:
                h_padding = -h_padding
                input = input[:, :, h_padding:-h_padding, :]
            else:
                input_h_padded = np.zeros(
                    (input.shape[0], input.shape[1], input.shape[2]+2*h_padding, input.shape[3]))
                input_h_padded[:, :, h_padding:-h_padding, :] = input
                input = input_h_padded

            # check if needs padding in width
            w_padding = self.w_filter - 1 - self.padding
            if w_padding == 0:
                pass
            elif w_padding < 0:
                w_padding = -w_padding
                input = input[:, :, w_padding:-w_padding, :]
            else:
                input_w_padded = np.zeros(
                    (input.shape[0], input.shape[1], input.shape[2], input.shape[3]+2*w_padding))
                input_w_padded[:, :, :, w_padding:-w_padding] = input
                input = input_w_padded

            # now calculate output shape
        self.out_height = max(0, input.shape[2] - self.h_filter + 1)
        self.out_width = max(0, input.shape[3] - self.w_filter + 1)
        if not self.out_width.is_integer() or not self.out_height.is_integer():
            raise Exception("[Tinynet] Invalid dimension settings!")
        self.output_shape = (input.shape[0], self.n_filter, self.out_height, self.out_width)
        self.input_in_col = im2col_indices(
            input, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding
        )
        # rotate the weight in 180 degree.
        weight_rot = np.flip(self.weight.tensor, (2,3))
        weight_in_row = weight_rot.reshape(self.n_filter, -1)
        output = np.matmul(weight_in_row, self.input_in_col) + self.bias.tensor
        output = output.transpose().reshape(*self.output_shape[1:], self.output_shape[0])
        output = output.transpose(3,0,1,2)
        return output
        
    def backward(self, in_gradient):
        gradient_flat = in_gradient.transpose(1,2,3,0).reshape(self.n_filter, -1)
        self.weight.gradient = np.matmul(gradient_flat, self.input_in_col.T).reshape(self.weight.tensor.shape).filp(2,3)
        self.bias.gradient = np.sum(in_gradient, axis=(0,2,3)).reshape(self.n_filter, -1)



