import tinyml
from tinyml.core import Backend as np
from tinyml.layers import Linear, ReLu
from tinyml.learner import Learner
from tinyml.losses import mse_loss
from tinyml.net import Sequential
from tinyml.optims import SGDOptimizer

tinyml.utilities.logger.VERBOSE = 1

X = np.random.randint(low=1, high=5, size=(5, 2))
Y = X[:, 0] * X[:, 1]
Y = Y.reshape(5, 1)

print(X.shape)
print(Y.shape)

model = Sequential([
    Linear('fc_1', 2, 10),
    ReLu('relu_1'),
    Linear('fc_2', 10, 1),
])

model.summary()

learner = Learner(model, mse_loss, SGDOptimizer(lr=0.3))

learner.fit(X, Y, epochs=5, batch_size=50)
