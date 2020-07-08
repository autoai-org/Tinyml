import numpy as np

def cross_entropy_loss(predicted, ground_truth):
    predicated_shift = predicted - np.max(predicted, axis=-1, keepdims=True)
    predicted_exp = np.exp(predicated_shift)
    predicated_probability = predicted_exp / np.sum(predicted_exp, axis=-1,keepdims=True)
    # print(predicated_probability.shape)
    # print(ground_truth.shape)
    loss = np.mean(np.sum(-ground_truth * np.log(predicated_probability+1e-15), axis=-1))
    return loss, predicated_probability - ground_truth