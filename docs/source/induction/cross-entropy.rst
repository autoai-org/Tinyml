Cross Entropy Loss
==================

The cross-entropy loss is defined as
:math:`l=-\sum_i^n \hat{y_i}log(p(y_i))` where :math:`p(y_i)` is the
probability of the output number, i.e. we usually use cross-entropy loss
after a softmax layer. By this nature, we could actually compute the
derivative of cross-entropy loss with respect to the original output
:math:`y_i` rather than :math:`p(y_i)`.

Then we have:

.. math:: \frac{\partial l}{\partial y_i}=- \sum_j \hat{y_j} \frac{\partial log(p(y_j))}{\partial y_i} = -\sum_j \hat{y_j} \frac{1}{p(y_j)}\frac{\partial p(y_j)}{\partial y_i}

Then as we know there will be a :math:`k=i` such that
:math:`\frac{p(y_k)}{\partial y_i}=p(y_j)(1-p(y_j))`, and for other
:math:`k\neq i`, we have
:math:`\frac{p(y_k)}{\partial y_i}=-p(y_j)p(y_i)`.

Then we have:

.. math::

   \begin{array}{l}
   -\sum_j \hat{y_j} \frac{1}{p(y_j)}\frac{\partial p(y_j)}{\partial y_i} \\ 
   = (-y_i)(1-p(y_i))-\sum_{j\neq i} \hat{y_j} \frac{1}{p(y_j)}p(y_j)p(y_i) \\
   = -y_i + p(y_i)y_i + \sum_{j\neq i}y_jp(y_i) \\ 
   = -y_i + p(y_i)\sum_{j\neq i} y_j \\ 
   = -y_i + p(y_i)\sum_{j}p(y_j) \\
   = p(y_i) - y_i
   \end{array}

The form is very elegant, and easy to compute. Therefore we usually hide
the computational process of the derivative of softmax in the
computation of cross entropy loss.

The implementation of cross entropy loss in Tinynet is as below:

.. literalinclude:: ../../../tinynet/losses/cross_entropy.py
  :language: Python
  :linenos:
