from .base import Net

from tinynet.core import Backend as np
from tinynet.utilities.logger import output_intermediate_result
from tinynet.layers import MaxPool2D

class Sequential(Net):
    '''
    Sequential model reads a list of layers and stack them to be a neural network.
    '''
    def __init__(self, layers):
        super().__init__(layers)
    
    def forward(self, input, return_indices=False):
        pool_indices = [None] * len(self.layers)
        output = input
        for index, layer in enumerate(self.layers):
            if isinstance(layer, MaxPool2D) and layer.return_index:
                output, max_indices = layer.forward(output)
                pool_indices[index] = max_indices
            else:
                output = layer.forward(output)
            output_intermediate_result(layer.name, output, 'data', layer)
        if return_indices:
            return output, pool_indices
        return output
    
    def backward(self, in_gradient):
        out_gradient = in_gradient
        for layer in self.layers[::-1]:    
            out_gradient = layer.backward(out_gradient)
            output_intermediate_result(layer.name, out_gradient, 'gradient', layer)
        return out_gradient

    def add(self, layer):
        self.layers.append(layer)

    def build_params(self):
        for layer in self.layers:
            self.parameters.extend(layer.parameters)

    def __call__(self, input):
        return self.forward(input)
    
    def predict(self, data, batch_size=None):
        results = None
        if batch_size:
            for i in range(0, len(data), batch_size):
                result = self.forward(data[i:i+batch_size])
                if results is None:
                    results = result
                else:
                    results = np.concatenate((results, result))
            return results
        else:
            return self.forward(data)