from .base import Layer
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, name):
        super().__init__(name)
    def forward(self, input):
        pass
    def backward(self, in_gradient):
        pass