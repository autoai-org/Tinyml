from .base import Layer
from tinyml.core import Backend as np

class Dropout(Layer):
    '''
    Dropout Layer randomly drop several nodes.
    '''
    def __init__(self, name, probability):
        super().__init__(name)
        self.probability = probability
        self.type = 'Dropout'
    def forward(self, input):
        self.mask = np.random.binomial(1, self.probability, size=input.shape) / self.probability
        return (input * self.mask).reshape(input.shape)
    
    def backward(self, in_gradient):
        return in_gradient * self.mask