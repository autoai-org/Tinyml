from .base import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self,name):
        super().__init__(name)
    
    def forward(self, input):
        self.input = input
        self.output_shape = (self.input.shape[0],-1)
        output = input.ravel().reshape(self.output_shape)
        return output
    
    def backward(self, in_gradient):
        out_gradient = in_gradient.reshape(self.input.shape)
        return out_gradient
