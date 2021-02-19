ReLu
====

The purpose of using activation functions is to bring some non-linearity
into the deep neural networks, so that the networks can fit the real
world. One of the most popular activation function is the
**re**\ ctifier **l**\ inear **u**\ nit (ReLu).

The function is defined as :math:`f(x)=max(0,x)`. Thus the forward pass
is simple: :math:`y_i=max(0, x_i)`.

In the ReLu function, we do not have any weight or bias to update. Hence
we only need to compute the gradient to previous layer. We have
:math:`\frac{\partial{l}}{\partial{x_i}}=\frac{\partial{l}}{\partial{y_i}}\frac{\partial{y_i}}{\partial{x_i}}`.

Then we have:

.. math::

   \frac{\partial{l}}{\partial{x_i}}=
     \begin{cases}
       0 & \text{$x_i$<0}  \\
       \frac{\partial{l}}{\partial{y_i}} & \text{$x_i$>0} \\
       undefined & \text{$x_i$=0}
     \end{cases}

We see that the derivative is not defined at the point :math:`x_i=0`,
but when computing, we can set it to be :math:`0`, or :math:`1`, or any other values between.

The implementation of ReLu layer in tinyml is as below:

.. literalinclude:: ../../../tinyml/layers/relu.py
  :language: Python
  :linenos: