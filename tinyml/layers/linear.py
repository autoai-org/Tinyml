from .base import Layer
from tinyml.core import Backend as np

class Linear(Layer):
    '''
    Linear layer performs fully connected operation.
    '''
    def __init__(self, name, input_dim, output_dim):
        super().__init__(name)
        # Xavier Initialization
        weight = np.random.randn(
            output_dim, input_dim) * np.sqrt(1/input_dim)
        bias = np.random.randn(output_dim) * np.sqrt(1/input_dim)
        self.type = 'Linear'
        self.weight = self.build_param(weight)
        self.bias = self.build_param(bias)
    
    def forward(self, input):
        '''
        The forward pass of fully connected layer is given by :math:`f(x)=wx+b`.
        '''
        # save input as the input will be used in backward pass
        self.input = input
        return np.matmul(input, self.weight.tensor.T) + self.bias.tensor
    
    def backward(self, in_gradient):
        '''
        In the backward pass, we compute the gradient with respect to :math:`w`, :math:`b`, and :math:`x`.

        We have:

        .. math::

            \\frac{\\partial l}{\\partial w} = \\frac{\\partial l}{\\partial y}\\frac{\\partial y}{\\partial w}=\\frac{\\partial l}{\\partial y} x
        '''
        self.weight.gradient = np.matmul(self.input.T, in_gradient).T
        self.bias.gradient = np.sum(in_gradient, axis=0)
        previous = np.matmul(in_gradient, self.weight.tensor)
        return np.matmul(in_gradient, self.weight.tensor)
    