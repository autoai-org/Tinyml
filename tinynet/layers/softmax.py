import numpy as np
from .base import Layer    

class Softmax(Layer):
    def __init__(self, name='softmax'):
        super().__init__(name)
    
    def forward(self, input):
        exps = np.exp(input - np.max(input))
        '''
        Some computational stability tricks here.
        '''
        logsumexp = np.log(np.sum(np.exp(input-np.max(input, axis=0)), axis=0))
        return np.exp(input - logsumexp)

    def backward(self, in_gradient):
        return in_gradient
