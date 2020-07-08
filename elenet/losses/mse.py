import numpy as np

def mse_loss(predicted, ground_truth):
    diff = predicted - ground_truth
    return np.square(diff).mean(), predicted - ground_truth