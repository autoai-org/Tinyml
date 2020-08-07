# The large end-to-end test. It is based on the mnist-cnn example.
import unittest

import numpy as np

import torch
import torch.nn as nn
from tests.base import EPSILON, isEqual
from tinynet.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLu
from tinynet.losses.cross_entropy import cross_entropy_with_softmax_loss
from tinynet.net import Sequential
from tinynet.optims import SGDOptimizer
from torch.nn import Conv2d as torch_conv2d
from torch.nn import Linear as torch_linear
from torch.nn import MaxPool2d as torch_maxpool2d
from torch.nn import ReLU as torch_relu


class torch_net(nn.Module):
    def __init__(self):
        super(torch_net, self).__init__()
        self.conv1 = torch_conv2d(3, 32, (3, 3), 1)
        self.relu1 = torch_relu()
        self.conv2 = torch_conv2d(32, 64, (3, 3), 1)
        self.relu2 = torch_relu()
        self.pool1 = torch_maxpool2d((2, 2), 2)
        self.linear1 = torch_linear(12 * 12 * 64, 128)
        self.relu3 = torch_relu()
        self.linear2 = torch_linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x


class End2EndTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tnn_model = Sequential([
            Conv2D('conv_1', (3, 28, 28),
                   n_filter=32,
                   h_filter=3,
                   w_filter=3,
                   stride=1,
                   padding=0),
            ReLu('relu_1'),
            Conv2D('conv_2', (32, 26, 26),
                   n_filter=64,
                   h_filter=3,
                   w_filter=3,
                   stride=1,
                   padding=0),
            ReLu('relu_2'),
            MaxPool2D('maxpool_1', (64, 24, 24), size=(2, 2), stride=2),
            Flatten('flat_1'),
            Linear('fc_1', 9216, 128),
            ReLu('relu_3'),
            Linear('fc_2', 128, 10),
        ])
        self.tnn_model.build_params()
        self.torch_model = torch_net()

        self.torch_model.conv1.weight = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[0].weight.tensor))
        self.torch_model.conv1.bias = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[0].bias.tensor))

        self.torch_model.conv2.weight = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[2].weight.tensor))
        self.torch_model.conv2.bias = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[2].bias.tensor))

        self.torch_model.linear1.weight = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[6].weight.tensor))
        self.torch_model.linear1.bias = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[6].bias.tensor))
        self.torch_model.linear2.weight = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[8].weight.tensor))
        self.torch_model.linear2.bias = nn.Parameter(
            torch.from_numpy(self.tnn_model.layers[8].bias.tensor))

        self.torch_optimizer = torch.optim.SGD(self.torch_model.parameters(),
                                               lr=0.1, momentum=0.9)
        self.torch_loss = nn.CrossEntropyLoss()

        self.tnn_optimizer = SGDOptimizer(lr=0.1, momentum=0.9)

    def test(self):
        batch_size = 1
        self.data = np.random.randn(batch_size, 3, 28, 28)
        self.gt = np.random.randint(0, 9, size=(batch_size, ))
        self.torch_input = torch.from_numpy(self.data)
        self.torch_input.requires_grad = True

        epochs = 2
        for epoch in range(epochs):
            isEqual(epoch, self.tnn_model, self.torch_model)
            self.torch_model.train()
            self.torch_optimizer.zero_grad()

            self.torch_output = self.torch_model(self.torch_input)
            self.torch_output.retain_grad()
            self.tnn_output = self.tnn_model(self.data)

            self.assertTrue(
                (self.torch_output.detach().numpy() - self.tnn_output <
                 pow(10, epoch) * EPSILON).all())

            self.torch_loss_val = self.torch_loss(
                self.torch_output,
                torch.from_numpy(self.gt).long())

            self.tnn_loss, self.tnn_loss_gradient = cross_entropy_with_softmax_loss(
                self.tnn_output, self.gt)
                        
            self.assertTrue(
                (self.torch_loss_val.detach().numpy() - self.tnn_loss <
                 EPSILON).all())
                
            # now perform the backward process
            self.torch_loss_val.backward()
            self.torch_optimizer.step()

            self.assertTrue(
                np.absolute((self.torch_output.grad.numpy() -
                             self.tnn_loss_gradient) < EPSILON).all())

            self.tnn_model.backward(self.tnn_loss_gradient)
            self.tnn_model.update(self.tnn_optimizer)


if __name__ == '__main__':
    unittest.main()
