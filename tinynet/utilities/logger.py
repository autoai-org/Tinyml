#coding:utf-8
from tabulate import tabulate
import numpy as np
VERBOSE = 1

def print_net_summary(layers):
    total_weight = [np.prod(layer.weight.tensor.shape) for layer in layers if hasattr(layer, 'weight')]
    total_bias = [np.prod(layer.bias.tensor.shape) for layer in layers if hasattr(layer, 'bias')]
    layers_info = [each.summary() for each in layers]
    print(tabulate(layers_info, headers=['Type', 'Name', 'Weight','Output_shape'], tablefmt='orgtbl'))
    print("Total Trainable Params: {}".format(sum(total_weight)+sum(total_bias)))

def log_trainining_progress(epoch,total_epochs, loss_sum, loss_mean):
    if (VERBOSE>=1):
        print("[Tinynet] epoch: {}/{}, loss(sum): {}, loss(mean): {}".format(epoch + 1, total_epochs, loss_sum, loss_mean))
    else:
        pass

def log_training_time(time_in_seconds):
    if (VERBOSE >= 1):
        print("[Tinynet] Finished training in {} seconds".format(time_in_seconds))

def output_intermediate_result(layername, output, type, layer):
    if (VERBOSE >= 3):
        print('--- {} {} ---'.format(type, layername))
        print('> shape: '+str(output.shape))
        print(output)
        if layer is not None and hasattr(layer, 'weight'):
            print ('==== {} weight ===='.format(layername))
            print (layer.weight.tensor)
            print ('==== {} bias ===='.format(layername))
            print (layer.bias.tensor)
            print ('==== {} weight gradient ===='.format(layername))
            print (layer.weight.gradient)
            print ('==== {} bias gradient ===='.format(layername))
            print (layer.bias.gradient)
        print('--- end of data ---')
