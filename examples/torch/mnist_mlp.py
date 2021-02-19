import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tinyml.core import Backend as np

from torch.autograd import Variable
from tinyml.utilities.logger import log_trainining_progress
import tinyml.dataloaders.mnist as mnist
import tinyml


tinyml.utilities.logger.VERBOSE = 1

def process_y(y):
    digits = 10
    examples = y.shape[0]
    y = y.reshape(1, examples)
    y = np.eye(digits)[y.astype('int32')]
    y = y.T.reshape(examples,digits)
    return y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,  10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

print('loading data...')
# mnist.init()
x_train, y_train, x_test, y_test = mnist.load()

x_train = (x_train/255).astype('float32')
x_test = (x_test/255).astype('float32')

y_train = process_y(y_train)
y_test = process_y(y_test)

print(y_test.shape)

model = Net()
optimizer = optim.SGD(model.parameters(),lr=0.05)

epochs = 10
batch_size = 256

def train(model, optimizer, epochs, batch_size, data, label):
    losses = []

    for epoch in range(epochs):
        model.train()
        p = np.random.permutation(len(data))
        data, label = data[p], label[p]
        loss = 0.0
        for i in range(0, len(data), batch_size):
            x_minibatch = data[i:i+batch_size]
            y_minibatch = label[i:i+batch_size]
            optimizer.zero_grad()
            output = model(Variable(torch.from_numpy(x_minibatch)))
            loss = F.cross_entropy(output, Variable(torch.from_numpy(y_minibatch)))
            loss.backward()
            optimizer.step()
        log_trainining_progress(epoch, epochs, loss, loss/batch_size)
        losses.append(loss)
    return losses

train(model, optimizer, epochs, batch_size, x_train, y_train)