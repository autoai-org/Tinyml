import numpy as np

def cross_entropy_loss(predicted, ground_truth):

    # print(predicated_probability.shape)
    # print(ground_truth.shape)
    loss = np.mean(np.sum(-ground_truth * np.log(predicted+1e-15), axis=-1))
    return loss, predicted - ground_truth