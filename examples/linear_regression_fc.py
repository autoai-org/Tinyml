from tinyml.core import Backend as np
from tinyml.layers import LeakyReLu, Linear, ReLu
from tinyml.learner import Learner
from tinyml.losses import mse_loss
from tinyml.net import Sequential
from tinyml.optims import SGDOptimizer

X = np.random.randn(100, 10)
w = np.random.randn(10, 1)
b = np.random.randn(1)
Y = np.matmul(X, w) + b

model = Sequential([
    Linear('fc_1', 10, 1),
])
model.summary()
learner = Learner(model, mse_loss, SGDOptimizer(lr=0.05))
learner.fit(X, Y, epochs=10, batch_size=10)
