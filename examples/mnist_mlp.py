import numpy as np

from tinynet.net import Sequential
from tinynet.layers import Linear, Softmax, ReLu
from tinynet.optims import SGDOptimizer
from tinynet.utilities.learner import Learner
from tinynet.losses import cross_entropy_loss, mse_loss
import tinynet.dataloaders.mnist as mnist

print('loading data...')

# mnist.init()
x_train, y_train, x_test, y_test = mnist.load()

x_train = (x_train/255).astype('float32')
x_test = (x_test/255).astype('float32')

digits = 10
examples = y_train.shape[0]
y = y_train.reshape(1, examples)
y = np.eye(digits)[y.astype('int32')]
y_train = y.T.reshape(examples,digits)

print(y_train.shape)
print(x_train.shape)
print('building model...')

model = Sequential([
    Linear('fc_1', 784, 80),
    ReLu('relu_1'),
    Linear('fc_2', 80, 10)
])

model.summary()
learner = Learner(model, cross_entropy_loss, SGDOptimizer(lr=0.02))

print('starting training...')
learner.fit(x_train, y_train, epochs=1, batch_size=256)