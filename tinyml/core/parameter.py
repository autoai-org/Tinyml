from tinyml.core import Backend as np

class Parameter():
  def __init__(self, tensor, require_grad=True):
    self.tensor = tensor
    self.gradient = np.zeros_like(self.tensor)
    self.require_grad = require_grad
    self.velocity = 0
    
  def __repr__(self):
    return '\nValue: {}\n Gradient: {}'.format(self.tensor, np.sum(self.gradient))