from tinynet.core import Backend as np
from sklearn.metrics import log_loss
from tinynet.layers import Softmax

def cross_entropy_loss(predicted, ground_truth):
    '''
    compute the cross entropy loss
    '''
    print(predicted)
    loss = np.mean(np.sum(-ground_truth * np.log(predicted + 1e-10), axis=-1))
    return log_loss(ground_truth, predicted), predicted - ground_truth

def cross_entropy_with_softmax_loss(predicted, ground_truth):
    softmax = Softmax('softmax')
    output_probabilities = softmax(predicted)
    return np.sum(-ground_truth * np.log(output_probabilities + 1e-10)), predicted - ground_truth