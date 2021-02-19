from tinyml.core import Backend as np
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
        shifted = np.exp(input - input.max(axis=1, keepdims=True))
        result = shifted / shifted.sum(axis=1, keepdims=True)
        return result

    def backward(self, in_gradient):
        '''
        Important: The actual backward gradient is not :math:`1`.

        The reason why we pass the gradient directly to previous layer is: since we know the formula is pretty straightforward when softmax is being used together with cross entropy loss (see theoretical induction), we compute the gradient in the cross entropy loss function, so that we could reduce the complexity, and increase the computational stabilities.
        '''
        return in_gradient
