import numpy as np

class Parameter():
  def __init__(self, tensor):
    self.tensor = tensor
    self.gradient = np.zeros_like(self.tensor)
  
  def __repr__(self):
    return '\nValue: {}\n Gradient: {}'.format(self.tensor, np.sum(self.gradient))