from tinynet.utilities.logger import print_net_summary

class Net(object):
    def __init__(self, layers):
        self.layers = layers
        self.parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameters)
        
    def update(self, optimizer):
        '''
        updates the saved parameters with the given optimizer
        '''
        # print(self.parameters)
        for param in self.parameters:
            optimizer.update(param)
        # print(self.parameters)

    def summary(self):
        print_net_summary(self.layers)
