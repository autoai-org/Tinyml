import tinynet
from tinynet.core import Backend as np
from tinynet.layers import Linear, ReLu
from tinynet.learner import Learner
from tinynet.losses import mse_loss
from tinynet.net import Sequential
from tinynet.optims import SGDOptimizer

tinynet.utilities.logger.VERBOSE = 1

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
