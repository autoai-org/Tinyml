import numpy as np

class Parameter():
  def __init__(self, tensor):
    self.tensor = tensor
    self.gradient = np.zeros_like(self.tensor)