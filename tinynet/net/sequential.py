from .base import Net

import numpy as np

class Sequential(Net):
    def __init__(self, layers):
        super().__init__(layers)
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, in_gradient):
        out_gradient = in_gradient
        for layer in self.layers[::-1]:
            out_gradient = layer.backward(out_gradient)
            print('{} gradient: {}'.format(layer.name, np.mean(out_gradient)))
        return out_gradient
