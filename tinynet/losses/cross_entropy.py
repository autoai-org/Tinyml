from tinynet.core import Backend as np
from tinynet.layers import Softmax

def cross_entropy_with_softmax_loss(predicted, ground_truth):
    softmax = Softmax('softmax')
    output_probabilities = softmax(predicted)
    loss = np.mean(-np.log(output_probabilities[np.arange(output_probabilities.shape[0],dtype=np.int8), ground_truth]))
    output_probabilities[np.arange(output_probabilities.shape[0]), ground_truth] -= 1
    gradient = output_probabilities / predicted.shape[0]
    return loss, 