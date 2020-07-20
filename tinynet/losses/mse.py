from tinynet.core import Backend as np

def mse_loss(predicted, ground_truth):
    '''
    Compute the mean square error loss.
    '''
    diff = predicted - ground_truth.reshape(predicted.shape)
    print(diff.shape)
    return np.square(diff).mean(), 2 * diff / diff.shape[1]