import os
import pickle

import numpy

# Higher verbose level = more detailed logging
import tinynet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tinynet.core import Backend as np
from tinynet.learner import Learner
from tinynet.losses import cross_entropy_with_softmax_loss
from tinynet.models.tinyvgg16 import tinyvgg16
from tinynet.optims import SGDOptimizer

GPU = True

if GPU:
    os.environ['TNN_GPU'] = "True"

tinynet.utilities.logger.VERBOSE = 1


def load_data(filepath):
    with open(filepath, 'rb') as f:
        cat_dog_data = pickle.load(f)
        x_train = cat_dog_data['train']['data']
        y_train = cat_dog_data['train']['label']
        x_test = cat_dog_data['test']['data']
        y_test = cat_dog_data['test']['label']
        return numpy.asarray(x_train), numpy.asarray(y_train), numpy.asarray(
            x_test), numpy.asarray(y_test)


def get_accuracy(y_predict, y_true):
    return np.mean(
        np.equal(np.argmax(y_predict, axis=-1), np.argmax(y_true, axis=-1)))


x_train, y_train, x_test, y_test = load_data("./dataset/tinyimagenet.pkl")

print(y_train.shape)
print(x_train.shape)

if GPU:
    import cupy as cp
    x_train = cp.array(x_train)
    y_train = cp.array(y_train)
    x_test = cp.array(x_test)
    y_test = cp.array(y_test)

model = tinyvgg16()
model.summary()

learner = Learner(model, cross_entropy_with_softmax_loss,
                  SGDOptimizer(lr=0.01))

TRAIN = True

print('starting training...')

if TRAIN:
    learner.fit(x_train, y_train, epochs=10, batch_size=10)
    model.export('tinyimagenet.tnn')

else:
    model.load('tinyimagenet.tnn')

print('starting evaluating...')

y_predict = learner.predict(x_test, batch_size=1)

acc = get_accuracy(y_predict, y_test)
print('Testing Accuracy: {}%'.format(acc * 100))
