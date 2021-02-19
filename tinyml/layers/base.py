#coding:utf-8

from tinyml.core import Parameter

class Layer(object):
    def __init__(self, name):
        self.name = name
        self.parameters = []
        self.type = 'unknown'
    
    def forward(self, input):
        '''
        > this function needs to be overridden.
        compute the forward pass
        '''
        return self.input
    
    def build_param(self, tensor):
        '''
        creates a parameter from a tensor, and save to the parameters.
        '''
        param = Parameter(tensor)
        self.parameters.append(param)
        return param
    
    def backward(self, in_gradient):
        '''
        > this function needs to be overridden.
        '''
        return in_gradient

    def _rebuild_params(self):
        '''
        In case users changed the weight after initialization, they can use this function to rebuild the params. With this function, the gradient information will be attached to the original parameters.

        Use this function with caution.
        '''
        self.weight = self.build_param(self.weight)
        self.bias = self.build_param(self.bias)

    def summary(self):
        info = [self.type, self.name]
        if hasattr(self, 'weight'):
            info.append(self.weight.tensor.shape)
        else:
            info.append('N/A')
        if hasattr(self, 'out_dim'):
            info.append(self.out_dim)
        else:
            info.append('N/A')
        return info
    
    def __call__(self, *args):
        return self.forward(*args)