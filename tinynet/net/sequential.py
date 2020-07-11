from .base import Net

import numpy as np
from tinynet.utilities.logger import log_backward_gradient

class Sequential(Net):
    def __init__(self, layers):
        super().__init__(layers)
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, in_gradient):
        for layer in self.layers[::-1]:
            in_gradient = layer.backward(in_gradient)
            log_backward_gradient(layer.name, np.mean(in_gradient))
        return in_gradient
