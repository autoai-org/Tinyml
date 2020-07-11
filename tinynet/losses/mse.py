import numpy as np

def mse_loss(predicted, ground_truth):
    '''
    Compute the mean square error loss.
    '''
    diff = predicted - ground_truth.reshape(predicted.shape)
    return np.square(diff).mean(), 2 * diff / len(diff)