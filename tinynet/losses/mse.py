from tinynet.core import Backend as np

def mse_loss(predicted, ground_truth):
    '''
    Compute the mean square error loss.
    '''
    diff = predicted - ground_truth.reshape(predicted.shape)
    return (diff ** 2).mean(), 2 * diff / diff.shape[1]