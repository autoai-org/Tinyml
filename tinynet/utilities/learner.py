import numpy as np

class Learner():
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def batch_fit(self, data, label):
        y_predicated = self.model.forward(data)
        loss, loss_gradient = self.loss(y_predicated, label)
        self.model.backward(loss_gradient)
        self.model.update(self.optimizer)
        return loss

    def fit(self, data, label, epochs, batch_size):
        losses = []
        for epoch in range(epochs):
            # shuffle it first
            p = np.random.permutation(len(data))
            data, label = data[p], label[p]
            loss = 0.0
            for i in range(0, len(data), batch_size):
                loss += self.batch_fit(data[i:i+batch_size],
                                       label[i:i+batch_size])
            print("[tinynet] epoch: {}/{}, loss(sum): {}, loss(mean): {}".format(epoch + 1, epochs, loss, loss/batch_size))
            losses.append(loss)
        return losses

    def predict(self, data):
        return self.model.forward(data)