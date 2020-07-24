from tinynet.layers import Linear, ReLu
from tinynet.losses import mse_loss
from tinynet.net import Sequential
from tinynet.optims import SGDOptimizer
import numpy as np

import tinynet

tinynet.utilities.logger.VERBOSE = 3

x = np.array([0.25, 0.65]).reshape(1, 2)
y = np.array([0.99, 0.01]).reshape(1, 2)

print(x.shape)
print(y.shape)

fc_1 = Linear('fc_1', 2, 2)
fc_2 = Linear('fc_2', 2, 2)

relu = ReLu('relu_1')

fc_1.weight = np.array([0.20, 0.25, 0.30, 0.35]).reshape(2, 2)
fc_1.bias = np.array([0.30, 0.30])
fc_1._rebuild_params()

fc_2.weight = np.array([0.50, 0.55, 0.60, 0.65]).reshape(2, 2)
fc_2.bias = np.array([0.20, 0.20])
fc_2._rebuild_params()

print(fc_2.weight.tensor)

model = Sequential([fc_1, relu, fc_2])
optimizer = SGDOptimizer(0.1)
model.summary()

epoch = 1

for epoch in range(epoch):
    y_predicted = model.forward(x)
    loss, loss_gradient = mse_loss(y_predicted, y)
    print('>>>>')
    print(loss)
    print(loss_gradient)
    print('<<<<')
    model.backward(loss_gradient)
    model.update(optimizer)
