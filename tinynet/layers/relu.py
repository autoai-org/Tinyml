from .base import Layer
from tinynet.core import Backend as np

class ReLu(Layer):
    '''
    ReLu layer performs rectifier linear unit opertaion.
    '''
    def __init__(self, name):
        super().__init__(name)
        self.type='ReLu'

    def forward(self, input):
        '''
        In the forward pass, the output is defined as :math:`y=max(0, x)`.
        '''
        self.input = input
        return input * (input > 0)

    def backward(self, in_gradient):
        return in_gradient * (self.input > 0)