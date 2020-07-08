import numpy as np

from tinynet.net import Sequential
from tinynet.layers import Linear, Softmax, ReLu
from tinynet.optims import SGDOptimizer
from tinynet.utilities.learner import Learner
from tinynet.losses import cross_entropy_loss
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

print('building model...')
model = Sequential([
    Linear('fc_1', 784, 256),
    ReLu('relu_1'),
    Linear('fc_2', 256, 64),
    ReLu('relu_2'),
    Linear('fc_2', 64, 10),
    Softmax()
])
model.summary()
learner = Learner(model, cross_entropy_loss, SGDOptimizer(lr=0.05))
print('starting training...')
learner.fit(x_train, y_train, epochs=15, batch_size=128)