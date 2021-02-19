import tinyml
from tinyml.core import Backend as np
from tinyml.layers import Linear, ReLu
from tinyml.learner import Learner
from tinyml.losses import mse_loss
from tinyml.net import Sequential
from tinyml.optims import SGDOptimizer

tinyml.utilities.logger.VERBOSE = 1

X = np.random.randint(low=1, high=5, size=(20, 20))
Y = np.prod(X, axis=1)
Y.reshape(20, 1)

print(X.shape)
print(Y.shape)

model = Sequential([
    Linear('fc_1', 20, 128),
    # ReLu('relu_1'),
    Linear('fc_2', 128, 1),
])

model.summary()

learner = Learner(model, mse_loss, SGDOptimizer(lr=0.01))

learner.fit(X, Y, epochs=50, batch_size=10)
