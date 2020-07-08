class SGDOptimizer():
    def __init__(self, lr):
        self.lr = lr
    
    def update(self, param):
        param.tensor -= self.lr * param.gradient
        param.gradient.fill(0)