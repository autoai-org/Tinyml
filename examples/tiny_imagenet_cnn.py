import os
import pickle

import numpy

# Higher verbose level = more detailed logging
import tinynet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tinynet.core import Backend as np
from tinynet.learner import Learner
from tinynet.learner.callbacks import (evaluate_classification_accuracy,
                                       save_model)
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
        idx = numpy.random.permutation(len(x_train))
        x_train, y_train = numpy.asarray(x_train)[idx], numpy.asarray(
            y_train)[idx]
        x_test = cat_dog_data['test']['data']
        y_test = cat_dog_data['test']['label']
        return x_train, y_train, numpy.asarray(x_test), numpy.asarray(y_test)


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
callbacks = [evaluate_classification_accuracy, save_model]
cargs = (x_test, y_test)

learner = Learner(model, cross_entropy_with_softmax_loss,
                  SGDOptimizer(lr=0.01, momentum=0.9))

TRAIN = True

print('starting training...')

if TRAIN:
    learner.fit(x_train,
                y_train,
                epochs=50,
                batch_size=45,
                callbacks=callbacks,
                callbacks_interval=1,
                cargs=cargs)
    model.export('tinyimagenet.tnn')

else:
    model.load('tinyimagenet.tnn')

print('starting evaluating...')

y_predict = learner.predict(x_test, batch_size=1)

acc = get_accuracy(y_predict, y_test)
print('Testing Accuracy: {}%'.format(acc * 100))
