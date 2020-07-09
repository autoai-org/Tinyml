import numpy as np

def cross_entropy_loss(predicted, ground_truth):

    loss = loss = -1 * np.einsum('ij,ij->', ground_truth, np.log(predicted+1e-15), optimize=True) / ground_truth.shape[0]
    return loss, predicted - ground_truth