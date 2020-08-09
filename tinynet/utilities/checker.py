'''
This utility computes the gradient with finite differences
'''
import numpy as np

def gradient_check(input, function, eps=1e-15):
    gradient = np.zeros(input.shape)
    for i in range(len(input)):
        input[i] += eps
        fn_output_1 = function(input[i])
        input[i] -= 2 * eps
        fn_output_2 = function(input[i])
        gradient[i] = (fn_output_2 - fn_output_1) / (2 * eps)
        input[i] += eps
    return gradient
