from tinynet.core import Backend as np
from tinynet.layers import Softmax

def cross_entropy_with_softmax_loss(predicted, ground_truth):
    softmax = Softmax('softmax')
    output_probabilities = softmax(predicted)
    return np.sum(-ground_truth * np.log(output_probabilities + 1e-10)), predicted - ground_truth