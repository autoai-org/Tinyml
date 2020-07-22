from .base import Layer
from tinynet.core import Backend as np
from tinynet.core import GPU 

class Deconv2D(Layer):
    def __init__(self, name, input_dim, n_filter, h_filter, w_filter, stride, padding):
        pass