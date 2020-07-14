from tinynet.core import Backend as np
from tinynet.layers import Linear, ReLu, LeakyReLu
from tinynet.losses import mse_loss
from tinynet.optims import SGDOptimizer
from tinynet.learner import Learner
from tinynet.net import Sequential

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