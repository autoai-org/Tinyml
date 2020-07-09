import numpy as np

def mse_loss(predicted, ground_truth):
    '''
    Compute the mean square error loss.
    '''
    diff = predicted - ground_truth
    return np.square(diff).mean(), (predicted - ground_truth)