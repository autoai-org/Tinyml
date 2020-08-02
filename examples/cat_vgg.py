import os
import pickle
import sys

import numpy as np

# Higher verbose level = more detailed logging
import tinynet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tinynet.core import Backend as np
from tinynet.learner import Learner
from tinynet.learner.callbacks import (evaluate_classification_accuracy,
                                       save_model)
from tinynet.losses import cross_entropy_with_softmax_loss
from tinynet.models.vgg16 import vgg16
from tinynet.optims import SGDOptimizer

GPU = True

if GPU:
    os.environ['TNN_GPU'] = "True"

sys.path.insert(1, "/content/tinynet/")

tinynet.utilities.logger.VERBOSE = 1

print('loading data...')

# mnist.init()


# Utilities
def load_data(filepath):
    with open(filepath, 'rb') as f:
        cat_dog_data = pickle.load(f)
        data = cat_dog_data['image']
        label = cat_dog_data['labels']
        x_train, x_test, y_train, y_test = train_test_split(data,
                                                            label,
                                                            test_size=0.10,
                                                            random_state=42)
        return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data('dataset/cat_and_dog.pkl')

print(y_train.shape)
print(x_train.shape)

if GPU:
    import cupy as cp
    x_train = cp.array(x_train)
    y_train = cp.array(y_train)
    x_test = cp.array(x_test)
    y_test = cp.array(y_test)

model = vgg16()
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
                batch_size=5,
                callbacks=callbacks,
                callbacks_interval=1,
                cargs=cargs)

    model.export('cat_and_dog.tnn')

else:
    model.load('cat_and_dog.tnn')

print('starting evaluating...')

y_predict = learner.predict(x_test, batch_size=1)

