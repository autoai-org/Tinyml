import numpy as np
from sklearn.metrics import log_loss

def cross_entropy_loss(predicted, ground_truth):
    '''
    compute the cross entropy loss
    '''
    # log_likelihood = -np.log(predicted + 1e-15)
    # loss = np.sum(log_likelihood).mean()
    return log_loss(ground_truth, predicted), predicted - ground_truth
