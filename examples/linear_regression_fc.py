import numpy as np
from elenet.layers import Linear, ReLu, LeakyReLu
from elenet.losses import mse_loss
from elenet.optims import SGDOptimizer
from elenet.utilities.learner import Learner
from elenet.net import Sequential

X = np.random.randn(100, 10)
w = np.random.randn(10, 1)
b = np.random.randn(1)
Y = np.matmul(X, w)+ b

model = Sequential([
    Linear('fc_1', 10, 1),
])
model.summary()
learner = Learner(model, mse_loss, SGDOptimizer(lr=0.05))
learner.fit(X, Y, epochs=10, batch_size=10)