import pickle
from tinynet.utilities.logger import print_net_summary
from tinynet.core import Backend as np

class Net(object):
    def __init__(self, layers):
        self.layers = layers
        self.parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameters)
        
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
                layers_weights[layer.name] = layer.weight
                layers_bias[layer.name] = layer.bias
        layers_params = {'weight':layers_weights, 'bias':layers_bias}
        with open(filepath, 'wb') as export_file:
            pickle.dump(layers_params, export_file)
        print("[Tinynet] Successfully exported to {}".format(filepath))
    
    def load(self, filepath):
        layers_params = None
        with open(filepath, 'rb') as import_file:
            layers_params = pickle.load(import_file)
        layers_weights = layers_params['weight']
        layers_bias = layers_params['bias']
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                layer.weight = layers_weights[layer.name]
                layer.bias = layers_bias[layer.name]
        print("[Tinynet] Successfully imported from {}".format(filepath))