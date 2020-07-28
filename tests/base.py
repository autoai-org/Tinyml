'''
There might be small errors caused by computation of floating numbers and we cannot compare two floating values directly. If the errors are smaller than the EPSILON, we consider them to be the same values.
'''
EPSILON = 1e-15

'''
PyTorch uses autograd to compute the gradient, while Tinynet uses symbolic method. This nature will cause the results different slightly. If the differences are smaller than 1e-5, we consider them to be the same.
'''
GRAD_EPSILON=1e-5