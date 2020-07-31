from sklearn.preprocessing import OneHotEncoder

import tinynet
import tinynet.dataloaders.mnist as mnist
from tinynet.core import Backend as np
from tinynet.layers import Linear, ReLu, Softmax, softmax
from tinynet.learner import Learner
from tinynet.learner.callbacks import evaluate_classification_accuracy
from tinynet.losses import cross_entropy_with_softmax_loss
from tinynet.net import Sequential
from tinynet.optims import SGDOptimizer

# Higher verbose level = more detailed logging
tinynet.utilities.logger.VERBOSE = 1

print('loading data...')


# mnist.init()
def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y


x_train, y_train, x_test, y_test = mnist.load()
x_train, y_train, x_test, y_test = pre_process_data(x_train, y_train, x_test,
                                                    y_test)

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


def get_accuracy(y_predict, y_true):
    return np.mean(
        np.equal(np.argmax(y_predict, axis=-1), np.argmax(y_true, axis=-1)))


model.summary()
callbacks = [evaluate_classification_accuracy]
cargs = (x_test, y_test)
learner = Learner(model, cross_entropy_with_softmax_loss, SGDOptimizer(lr=0.5))

print('starting training...')
model, losses = learner.fit(x_train,
                            y_train,
                            epochs=2,
                            batch_size=256,
                            callbacks=callbacks,
                            callbacks_interval=1,
                            cargs=cargs)

# model.load("./model.tnn")

print('starting evaluating...')

y_predict = learner.predict(x_test, 100)
print(y_predict.shape)
acc = get_accuracy(y_predict, y_test)
print('Testing Accuracy: {}%'.format(acc * 100))
