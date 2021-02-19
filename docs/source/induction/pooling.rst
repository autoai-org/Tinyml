Max Pooling Layer
=================

Pooling layer is another important component of convolutional neural
networks. There are many ways of performing pooling in this layer, such
as max pooling, average pooling, etc. In this part, we will only discuss
max-pooling layer as it is used most commonly in convolutional neural
networks.

In Max pooling layer, we also have a spatially small sliding window
called the kernel. In the window, only the largest value will be
remained and all other values will be dropped. For example, assume we
have

.. math::

   A=\left[ {\begin{array}{*{20}c} 
           1 & 2 & 3 \\
           4 & 5 & 6   \\
           7 & 8 & 9   
       \end{array} } \right]

\ and a :math:`2\times 2` max-pooling kernel. Then the output :math:`C`
will be

.. math::

   C=\left[ {\begin{array}{*{20}c} 
           5 & 6 \\
           8 & 9 \\
       \end{array} } \right]

With the given kernel size :math:`K_w` and :math:`K_h`, We can formalize
the max-pooling process as

.. math::

   f(x_{ij})=
       \begin{cases}
           x_{ij} & \text{$x_{ij}\geq x_{mn}, \forall m\in [i-K_w, i+K_w], n\in [j-K_h,j+K_h]$} \\
           0      & \text{otherwise}                                                            \\
       \end{cases}

Hence we can compute the derivative as below:

.. math::

   \frac{\partial l}{\partial x_{ij}}=\frac{\partial l}{\partial f}\frac{\partial f}{\partial x_{ij}}=
       \begin{cases}
           \frac{\partial l}{\partial f} & \text{$x_{ij}\geq x_{mn}, \forall m\in [i-K_w, i+K_w], n\in [j-K_h,j+K_h]$} \\
           0                             & \text{otherwise}                                                            \\
       \end{cases}

The implementation of max pooling layer in tinyml is as below:

.. literalinclude:: ../../../tinyml/layers/pooling.py
  :language: Python
  :linenos: