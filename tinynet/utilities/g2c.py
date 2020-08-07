'''
This script converts a model trained on gpu to a model on cpu
'''
import cupy as cp
import numpy as np
import pickle
from tinynet.models.vgg11 import vgg11

def convert(filepath):
    model = vgg11()
    model.load(filepath)
    for layer in model.layers:
        if hasattr(layer, 'weight'):
            layer.weight = cp.asnumpy(layer.weight)
            layer.bias = cp.asnumpy(layer.bias)
    model.export('exported_.tnn')