import numpy as np

class Parameter():
  def __init__(self, tensor, require_grad=True):
    self.tensor = tensor
    self.gradient = np.zeros_like(self.tensor)
    self.require_grad = require_grad
    
  def __repr__(self):
    return '\nValue: {}\n Gradient: {}'.format(self.tensor, np.sum(self.gradient))