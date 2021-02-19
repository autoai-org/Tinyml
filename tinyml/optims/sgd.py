class SGDOptimizer():
    '''
    In this class, we implement the stochastic gradient descent algorithm, which is used to update the parameters in a neural network. The algorithm is simple:

    .. math::

        w^{new} = w^{old}-\\lambda \\nabla
    
    where :math:`\\lambda` is the preset learning rate, and :math:`\\nabla` is the gradient.
    '''
    def __init__(self, lr, momentum=None):
        self.lr = lr
        self.momentum = momentum
    
    def update(self, param):
        if param.require_grad:
            if self.momentum:
                param.velocity = self.momentum * param.velocity + self.lr * param.gradient
                param.tensor -= param.velocity
            else:
                param.tensor -= self.lr * param.gradient
            param.gradient.fill(0)
        else:
            pass