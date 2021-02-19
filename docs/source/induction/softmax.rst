Softmax
=======

Softmax is such a function that takes the output of the fully connected
layers, and turn them into the probability. Formally, it takes an
:math:`n`-d vector, and normalizes it to :math:`n` probabilities
proportional to the exponentials of the input number. It is defined as

.. math:: f(x)=\frac{e^{x_i}}{\sum e^{x_j}}

, where :math:`x_i` is the :math:`i`-th input number.

We can then compute the derivative by using the quotient rule (if
:math:`f(x)=\frac{g(x)}{h(x)}`, then
:math:`f'(x)=\frac{g'(x)h(x)-g(x)h(x)}{h^2(x)}`). In our case, we have
:math:`g_i=e^{x_i}` and :math:`h_i=\sum e^{x_j}`. Then we have
:math:`\frac{\partial g_i}{x_j}=e^{x_i} \: (i=j)` or
:math:`0 \: (i\neq j)`. For :math:`h_i`, no matter the relation between
:math:`i` and :math:`j`, the derivative will always be :math:`e^{x_i}`.

Thus we have:

When :math:`i=j`,

.. math:: \frac{\partial f}{\partial x_i}=\frac{e^{x_i}\sum e^{x_j}-e^{x_i}e^{x_j}}{(\sum e^{x_j})^2}=\frac{e^{x_i}}{\sum{e^{x_j}}}\times \frac{(\sum e^{x_i} - e^{x_i})}{\sum{e^{x_j}}} = f(x_i)(1-f(x_i))

When :math:`i\neq j`,

.. math:: \frac{\partial f}{\partial x_i}=\frac{0-e^{x_i}e^{x_j}}{(\sum e^{x_j})^2}=-\frac{e^{x_i}}{\sum e^{x_j}}\times \frac{e^{x_j}}{\sum e^{x_j}}=-f(x_i)f(x_j)

The implementation of softmax layer in tinyml is as below:

.. literalinclude:: ../../../tinyml/layers/softmax.py
  :language: Python
  :linenos: