import pickle
from tinyml.utilities.logger import print_net_summary
from tinyml.core import Backend as np
from tinyml.core import GPU
import numpy

class Net(object):
    def __init__(self, layers):
        self.layers = layers
        self.parameters = []
        
    def update(self, optimizer):
        '''
        updates the saved parameters with the given optimizer.
        '''
        for param in self.parameters:
            optimizer.update(param)

    def summary(self):
        print_net_summary(self.layers)
    
    def export(self, filepath):
        layers_weights = {}
        layers_bias = {}
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                if (GPU):
                    import cupy as cp
                    layers_weights[layer.name] = cp.asnumpy(layer.weight.tensor)
                    layers_bias[layer.name] = cp.asnumpy(layer.bias.tensor)
                else:
                    layers_weights[layer.name] = layer.weight.tensor
                    layers_bias[layer.name] = layer.bias.tensor
        layers_params = {'weight':layers_weights, 'bias':layers_bias}
        numpy.save(filepath, layers_params)
        print("[tinyml] Successfully exported to {}".format(filepath))
    
    def load(self, filepath):
        if filepath.endswith('.npy'):
            self._load_npy(filepath)
        else:
            layers_params = None
            with open(filepath, 'rb') as import_file:
                layers_params = pickle.load(import_file)
            layers_weights = layers_params['weight']
            layers_bias = layers_params['bias']
            for layer in self.layers:
                if hasattr(layer, 'weight'):
                    print(layers_weights[layer.name])
                    layer.weight.tensor = layers_weights[layer.name]
                    layer.bias.tensor = layers_bias[layer.name]
            print("[tinyml] Successfully imported from {}".format(filepath))
    
    def _load_npy(self, filepath):
        layers_params = np.load(filepath, allow_pickle=True)
        layers_weight = layers_params.item().get('weight')
        layers_bias = layers_params.item().get('bias')
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                layer.weight.tensor = layers_weight[layer.name]
                layer.bias.tensor = layers_bias[layer.name]
        print("[tinyml] Successfully imported from {}".format(filepath))
