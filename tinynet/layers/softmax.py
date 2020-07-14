from tinynet.core import Backend as np
from .base import Layer    

class Softmax(Layer):
    '''
    Softmax layer returns the probability proportional to the exponentials of the input number.
    '''
    def __init__(self, name='softmax', axis=1, eps=1e-10):
        super().__init__(name)
        self.type = 'Softmax'
        self.axis = 1
        self.eps=eps

    def forward(self, input):
        '''
        Some computational stability tricks here.
        > TODO: to add the tricks
        '''
        self.input = input
        input_max = np.max(input, axis=self.axis, keepdims=True)
        if input_max.ndim > 0:
            input_max[~np.isfinite(input_max)] = 0
        shifted_input = np.exp(input-input_max)
        sum_input = np.sum(shifted_input,axis=self.axis, keepdims=True)      
        logsumexp = np.log(sum_input + self.eps)
        logsumexp += input_max
        return np.exp(input-logsumexp)

    def backward(self, in_gradient):
        '''
        Important: The actual backward gradient is not :math:`1`.
        The reason why we pass in_gradient directly to previous layer is: since we know the formula is pretty straightforward when softmax is being used together with cross entropy loss (see theoretical induction), we compute the gradient in the cross entropy loss function, so that we could reduce the complexity, and increase the computational stabilities.
        '''
        return in_gradient
