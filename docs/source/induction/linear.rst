Fully Connected Layers
======================

In forward pass, the output of fully connected layers is simple:
:math:`y_i=w_i \times x_i + b_i`.

Then in order to know how :math:`w` changes will affect the loss, we
need to calculate :math:`\frac{\partial{l}}{\partial{w_i}}`. By using
the chain rule, we have
:math:`\frac{\partial{l}}{\partial{w_i}}=\frac{\partial{l}}{y_i}\frac{\partial{y_i}}{\partial{w_i}}=\frac{\partial{l}}{y_i}x_i`,
and
:math:`\frac{\partial{l}}{\partial{b_i}}=\frac{\partial{l}}{y_i}\frac{\partial{y_i}}{\partial{b_i}}=\frac{\partial{l}}{y_i}`.
We can then successfully update our weight and bias in this layer.

After updating the weight and bias in :math:`i`-th layer, we also need
to pass the gradient of loss with respect to the input to the previous
layer. So we need to compute the gradient that the :math:`i`-th layer
passed to previous layer by
:math:`\frac{\partial{l}}{\partial{x_i}}=\frac{\partial{l}}{\partial{y_i}}\frac{\partial{y_i}}{\partial{x_i}}=\frac{\partial{l}}{\partial{y_i}}w_i`.

The implementation of fully connected layer in Tinynet is as below:

.. literalinclude:: ../../../tinynet/layers/linear.py
  :language: Python
  :linenos:
