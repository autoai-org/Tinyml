import os
os.environ['TNN_GPU'] = ""
from tinynet.layers.flatten import Flatten
from tinynet.layers import softmax
from tinynet.core import Backend as np
from tinynet.net import Sequential
from tinynet.layers import Linear, Softmax, ReLu, Conv2D
from tinynet.optims import SGDOptimizer
from tinynet.learner import Learner
from tinynet.losses import cross_entropy_with_softmax_loss, mse_loss
import tinynet.dataloaders.mnist as mnist

from sklearn.preprocessing import OneHotEncoder

import tinynet

# Higher verbose level = more detailed logging
tinynet.utilities.logger.VERBOSE = 1

print('loading data...')
# mnist.init()


def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.
    train_x = train_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y

x_train, y_train, x_test, y_test = mnist.load()
x_train, y_train, x_test, y_test = pre_process_data(x_train, y_train, x_test, y_test)

print(y_train.shape)
print(x_train.shape)
print('building model...')

model = Sequential([
    Conv2D('conv_1', (1, 28, 28), n_filter=32,
           h_filter=3, w_filter=3, stride=1, padding=1),
    ReLu('relu_1'),
    Flatten('flatten_1'),
    Linear('fc_1', 25088, 256),
    ReLu('relu_2'),
    Linear('fc_2', 256, 64),
    ReLu('relu_3'),
    Linear('fc_2', 64, 10),
])


def get_accuracy(y_predict, y_true):
    return np.mean(np.equal(np.argmax(y_predict, axis=-1),
                            np.argmax(y_true, axis=-1)))


model.summary()
learner = Learner(model, cross_entropy_with_softmax_loss, SGDOptimizer(lr=0.01))

print('starting training...')
learner.fit(x_train, y_train, epochs=5, batch_size=1024)

print('starting evaluating...')

y_predict = learner.predict(x_test)

acc = get_accuracy(y_predict, y_test)
print('Testing Accuracy: {}%'.format(acc*100))
