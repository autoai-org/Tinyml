import numpy as np
from .base import Layer

class Softmax(Layer):
    def __init__(self, name='softmax'):
        super().__init__(name)
    
    def forward(self, input):
        self.input = input
        input -= np.max(input)
        softmax_output = (np.exp(input).T/np.sum(np.exp(input), axis=0)).T
        # save the output for backward pass
        self.output = softmax_output
        return softmax_output
    
    def backward(self, in_gradient):
        reshaped_input = self.output.reshape(-1,1)
        return np.diagflat(reshaped_input) - np.matmul(reshaped_input, reshaped_input.T)
