from .base import Net

from tinynet.core import Backend as np
from tinynet.utilities.logger import output_intermediate_result

class Sequential(Net):
    '''
    Sequential model reads a list of layers and stack them to be a neural network.
    '''
    def __init__(self, layers):
        super().__init__(layers)
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
            output_intermediate_result(layer.name, output, 'data')
        return output
    
    def backward(self, in_gradient):
        for layer in self.layers[::-1]:
            in_gradient = layer.backward(in_gradient)
            output_intermediate_result(layer.name, in_gradient, 'gradient')
        return in_gradient
    
    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, input):
        return self.forward(input)