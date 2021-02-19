import os

import tinyml
import tinyml.dataloaders.mnist as mnist
from sklearn.preprocessing import OneHotEncoder
from tinyml.core import Backend as np
from tinyml.layers import Conv2D, Dropout, Linear, ReLu, Softmax, softmax
from tinyml.layers.flatten import Flatten
from tinyml.layers.pooling import MaxPool2D
from tinyml.learner import Learner
from tinyml.learner.callbacks import evaluate_classification_accuracy
from tinyml.losses import cross_entropy_with_softmax_loss, mse_loss
from tinyml.net import Sequential
from tinyml.optims import SGDOptimizer

# Higher verbose level = more detailed logging
tinyml.utilities.logger.VERBOSE = 1

GPU = False

if GPU:
    os.environ['TNN_GPU'] = "True"

print('loading data...')
# mnist.init()


def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    return train_x, train_y, test_x, test_y


x_train, y_train, x_test, y_test = mnist.load()
x_train, y_train, x_test, y_test = pre_process_data(x_train, y_train, x_test,
                                                    y_test)

if GPU:
    import cupy as cp
    x_train = cp.array(x_train)
    y_train = cp.array(y_train)
    x_test = cp.array(x_test)
    y_test = cp.array(y_test)

print(y_train.shape)
print(x_train.shape)
print('building model...')

model = Sequential([
    Linear('fc_1', 784, 128),
    ReLu('relu_1'),
    Linear('fc_2', 128, 64),
    ReLu('relu_2'),
    Linear('fc_3', 64, 10),
])

model.build_params()

model.summary()
callbacks = [evaluate_classification_accuracy]
cargs = (x_test, y_test)
learner = Learner(model, cross_entropy_with_softmax_loss,
                  SGDOptimizer(lr=0.1, momentum=0.9))

print('starting training...')
learner.fit(x_train,
            y_train,
            epochs=5,
            batch_size=1024,
            callbacks=callbacks,
            callbacks_interval=1,
            cargs=cargs)

print('training completed!')
