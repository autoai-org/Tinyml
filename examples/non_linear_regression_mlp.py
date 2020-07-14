from tinynet.core import Backend as np
import tinynet

from tinynet.layers import Linear, ReLu
from tinynet.losses import mse_loss
from tinynet.optims import SGDOptimizer
from tinynet.learner import Learner
from tinynet.net import Sequential

tinynet.utilities.logger.VERBOSE = 3

X = np.random.randint(low=1, high=5, size=(5, 2))
Y = X[:, 0] * X[:, 1]
Y = Y.reshape(5,1)

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
