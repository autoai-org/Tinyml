'''
There might be small errors caused by computation of floating numbers and we cannot compare two floating values directly. If the errors are smaller than the EPSILON, we consider them to be the same values.
'''
EPSILON = 1e-10
'''
PyTorch uses autograd to compute the gradient, while Tinynet uses symbolic method. This nature will cause the results different slightly. If the differences are smaller than 1e-5, we consider them to be the same.
'''
GRAD_EPSILON = 1e-5

import numpy as np

from tests.base import EPSILON
from tinynet.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLu
from torch.nn import Conv2d as torch_conv2d
from torch.nn import Linear as torch_linear
from torch.nn import MaxPool2d as torch_maxpool2d
from torch.nn import ReLU as torch_relu


# This function is only used for e2e tests, and cannot be used directly for comparing two models.
def isEqual(epoch, tnn_model, torch_model):
    print('Epoch {}'.format(epoch))
    assert (np.absolute(torch_model.linear2.weight.detach().numpy() -
                        tnn_model.layers[8].weight.tensor < EPSILON)).all()

    assert (np.absolute(torch_model.linear2.bias.detach().numpy() -
                        tnn_model.layers[8].bias.tensor) < EPSILON).all()

    print('----- Linear 1 -----')
    print('----- BIAS -----')
    print(tnn_model.layers[6].bias.tensor)
    print('--------------')
    print(torch_model.linear1.bias.detach().numpy())
    '''
    assert (np.absolute(torch_model.linear1.weight.detach().numpy() -
                        tnn_model.layers[6].weight.tensor < EPSILON)).all()
    assert (np.absolute(torch_model.linear1.bias.detach().numpy() -
                        tnn_model.layers[6].bias.tensor) < EPSILON).all()

    assert (torch_model.conv2.weight.detach().numpy() -
            tnn_model.layers[2].weight.tensor < EPSILON).all()
    assert (torch_model.conv2.bias.detach().numpy() -
            tnn_model.layers[2].bias.tensor < EPSILON).all()

    assert (torch_model.conv1.weight.detach().numpy() -
            tnn_model.layers[0].weight.tensor < EPSILON).all()
    assert (torch_model.conv1.bias.detach().numpy() -
            tnn_model.layers[0].bias.tensor < EPSILON).all()
    '''
