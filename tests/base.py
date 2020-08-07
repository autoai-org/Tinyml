'''
There might be small errors caused by computation of floating numbers and we cannot compare two floating values directly. If the errors are smaller than the EPSILON, we consider them to be the same values.
'''
EPSILON = 1e-10

import numpy as np

from tests.base import EPSILON


# This function is only used for e2e tests, and cannot be used directly for comparing two models.
def isEqual(epoch, tnn_model, torch_model):
    assert (np.absolute(torch_model.linear2.weight.detach().numpy() -
                        tnn_model.layers[8].weight.tensor < EPSILON)).all()

    assert (np.absolute(torch_model.linear2.bias.detach().numpy() -
                        tnn_model.layers[8].bias.tensor) < EPSILON).all()

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
