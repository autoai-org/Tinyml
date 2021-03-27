from tinyml.models.inv_vgg11 import forward

import numpy as np

import torch
from tests.base import EPSILON
from tinyml.layers import Conv2D, MaxPool2D, Flatten, Linear, Deconv2D, MaxUnpool2D
from tinyml.net import Sequential
from tinyml.losses.cross_entropy import cross_entropy_with_softmax_loss
from tinyml.optims import SGDOptimizer

tnn_model = Sequential([
    Conv2D('test_conv2d', (1, 5, 5), 1, 2, 2, 1, 0),
    MaxPool2D('pool', (1, 4, 4), (2, 2), 1, return_index=True),
    Flatten('flat_1'),
    Linear('fc_1', 9, 2),
])

tnn_model.build_params()

data = np.array([[0.10, 0.15, 0.20, 0.25,
                  0.30], [0.35, 0.40, 0.45, 0.55, 0.60],
                 [0.65, 0.70, 0.75, 0.80,
                  0.85], [0.90, 0.95, 0.10, 0.15, 0.20],
                 [0.25, 0.30, 0.35, 0.40, 0.45]]).reshape(1, 1, 5, 5)

ground_truth = np.array([1])
forward_conv_weight = np.array([[0.1, 0.2], [0.3, 0.4]]).reshape(1, 1, 2, 2)

tnn_model.layers[0].weight.tensor = forward_conv_weight
tnn_model.layers[0].bias.tensor = np.zeros_like(
    tnn_model.layers[0].bias.tensor)

forward_fc_weight = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                              [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])

tnn_model.layers[3].weight.tensor = forward_fc_weight
tnn_model.layers[3].bias.tensor = np.zeros_like(
    tnn_model.layers[3].bias.tensor)

output = tnn_model(data)

tnn_loss, tnn_loss_gradient = cross_entropy_with_softmax_loss(
    output, ground_truth)

print(output)
print(tnn_loss)
print(tnn_loss_gradient)

tnn_optimizer = SGDOptimizer(lr=0.01)
#tnn_model.backward(tnn_loss_gradient)
#tnn_model.update(tnn_optimizer)

print(tnn_model.layers[3].weight.tensor)
print(tnn_model.layers[0].weight.tensor)

new_output = tnn_model(data)
tnn_loss, tnn_loss_gradient = cross_entropy_with_softmax_loss(
    new_output, ground_truth)

print(new_output)
print(tnn_loss)
print(tnn_loss_gradient)

feature_maps = tnn_model.layers[0](data)
feature_maps, indices = tnn_model.layers[1](feature_maps)

tnn_unpooling = MaxUnpool2D('unpool', (1, 3, 3), (2, 2), 1)
tnn_deconv = Deconv2D('test_deconv', (1, 4, 4), 1, 2, 2, 1, 1, 0)
tnn_deconv.weight.tensor = tnn_model.layers[0].weight.tensor

unpooled = tnn_unpooling(feature_maps, indices)
print(unpooled)
deconved = tnn_deconv(unpooled)
print(deconved)
