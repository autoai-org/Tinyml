from tinyml.models.inv_vgg11 import forward

import numpy as np

import torch
from tests.base import EPSILON
from tinyml.layers import Conv2D, MaxPool2D, Flatten, Linear
from tinyml.net import Sequential
from tinyml.losses.cross_entropy import cross_entropy_with_softmax_loss
from tinyml.optims import SGDOptimizer

tnn_model = Sequential([
    Conv2D('test_conv2d', (1, 3, 3), 1, 2, 2, 1, 0),
    Flatten('flat_1'),
    Linear('fc_1', 4, 2),
])

tnn_model.build_params()

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).reshape(1, 1, 3, 3)

ground_truth = np.array([0])
forward_conv_weight = np.array([[1., 2.], [3., 4.]]).reshape(1, 1, 2, 2)

tnn_model.layers[0].weight.tensor = forward_conv_weight
tnn_model.layers[0].bias.tensor = np.zeros_like(
    tnn_model.layers[0].bias.tensor)

forward_fc_weight = np.array([[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])

tnn_model.layers[2].weight.tensor = forward_fc_weight
tnn_model.layers[2].bias.tensor = np.zeros_like(
    tnn_model.layers[2].bias.tensor)

output = tnn_model(data)

tnn_loss, tnn_loss_gradient = cross_entropy_with_softmax_loss(
    output, ground_truth)

print(output)
print(tnn_loss)
print(tnn_loss_gradient)

tnn_optimizer = SGDOptimizer(lr=0.01)
tnn_model.backward(tnn_loss_gradient)
tnn_model.update(tnn_optimizer)

print(tnn_model.layers[2].weight.tensor)
print(tnn_model.layers[0].weight.tensor)

new_output = tnn_model(data)
tnn_loss, tnn_loss_gradient = cross_entropy_with_softmax_loss(
    new_output, ground_truth)

print(new_output)
print(tnn_loss)
print(tnn_loss_gradient)
