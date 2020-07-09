from .base import Layer
import numpy as np


class Linear(Layer):
    def __init__(self, name, input_dim, output_dim):
        super().__init__(name)
        weights = np.random.randn(
            input_dim, output_dim) * np.sqrt(1/input_dim)
        bias = np.zeros(output_dim)
        self.type = 'Linear'
        self.weights = self.build_param(weights)
        self.bias = self.build_param(bias)

    def forward(self, input):
        # save input as the input will be used in backward pass
        self.input = input
        return np.matmul(input, self.weights.tensor) + self.bias.tensor

    def backward(self, in_gradient):
        self.weights.gradient += np.matmul(self.input.T, in_gradient)
        self.bias.gradient += in_gradient.sum(axis=0)
        return np.matmul(in_gradient, self.weights.tensor.T)
