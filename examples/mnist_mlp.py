import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from elenet.net import Sequential
from elenet.layers import Linear, Softmax, ReLu
from elenet.optims import SGDOptimizer
from elenet.utilities.learner import Learner
from elenet.losses import cross_entropy_loss

x, y = fetch_openml('mnist_784', version=1, return_X_y = True, cache=True)
x = (x/255).astype('float32')

digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
y = np.eye(digits)[y.astype('int32')]
y = y.T.reshape(examples,digits)

model = Sequential([
    Linear('fc_1', 784, 256),
    ReLu('relu_1'),
    Linear('fc_2', 256, 256),
    ReLu('relu_2'),
    Linear('fc_3', 256, 10),
])
model.summary()
learner = Learner(model, cross_entropy_loss, SGDOptimizer(lr=0.1))
learner.fit(x, y, epochs=10, batch_size=32)