from .base import Layer
from tinynet.core import Backend as np

class Flatten(Layer):
    '''
    Flatten layer reads an ndarray as input, and reshape it into a 1-d vector.
    '''
    def __init__(self, name):
        super().__init__(name)
        self.type = 'Flatten'
    def forward(self, input):
        self.input = input
        self.output_shape = (self.input.shape[0],-1)
        output = input.ravel().reshape(self.output_shape)
        self.out_dim = output.shape
        return output
    
    def backward(self, in_gradient):
        out_gradient = in_gradient.reshape(self.input.shape)
        return out_gradient
