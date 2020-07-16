from tinynet.core import Backend as np
import time
from tinynet.utilities.logger import log_trainining_progress, log_training_time

class Learner():
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
    
    def batch_fit(self, data, label):
        y_predicated = self.model.forward(data)
        loss, loss_gradient = self.loss(y_predicated, label)
        loss, loss_gradient = loss/len(data), loss_gradient/len(data)
        self.model.backward(loss_gradient)
        self.model.update(self.optimizer)
        return loss

    def fit(self, data, label, epochs, batch_size):
        losses = []
        start = time.process_time()
        for epoch in range(epochs):
            # randomly shuffle the data and label.
            p = np.random.permutation(len(data))
            data, label = data[p], label[p]
            loss = 0.0
            # actual training in the mini-batch
            for i in range(0, len(data), batch_size):
                loss += self.batch_fit(data[i:i+batch_size],
                                       label[i:i+batch_size])
            log_trainining_progress(epoch, epochs, loss, loss/batch_size)
            losses.append(loss)
        elapsed = time.process_time() - start
        log_training_time(elapsed)
        return self.model, losses

    def predict(self, data, batch_size = None):
        results = None
        if batch_size:
            for i in range(0, len(data), batch_size):
                result = self.model.forward(data[i:i+batch_size])
                if results is None:
                    results = result
                else:
                    results = np.concatenate((results, result))
            return results
        else:
            return self.model.forward(data)