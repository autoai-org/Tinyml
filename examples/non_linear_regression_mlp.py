import numpy as np
from elenet.layers import Linear, ReLu, LeakyReLu
from elenet.losses import mse_loss
from elenet.optims import SGDOptimizer
from elenet.utilities.learner import Learner
from elenet.net import Sequential

X = np.random.randn(1000, 2)
Y = X[:, 0] * X[:, 1]
Y = Y.reshape(1000,1)

model = Sequential([
    Linear('fc_1', 2, 10),
    ReLu('relu_1'),
    Linear('fc_2', 10, 1),
])

model.summary()

learner = Learner(model, mse_loss, SGDOptimizer(lr=0.3))

learner.fit(X, Y, epochs=3, batch_size=2)
