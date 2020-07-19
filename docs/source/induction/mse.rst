Mean Square Loss
================

The mean square error is defined as
:math:`l = \frac{1}{n}\sum (y_i-\hat{y}^i)^2`. Since this is the last
derivative we need to compute, we will only need to compute
:math:`\frac{\partial l}{\partial y_i}`. Let
:math:`g(y_i)=y_i-\hat{y_i}`, then
:math:`\frac{\partial g}{\partial y_i}=1`.

.. math:: \frac{\partial l}{\partial y_i}=\frac{\partial l}{\partial g}\times \frac{\partial g}{{\partial y_i}}=\frac{2}{n}(y_i-\hat{y_i})

The implementation of mean square error loss in Tinynet is as below:

.. literalinclude:: ../../../tinynet/losses/mse.py
  :language: Python
  :linenos: