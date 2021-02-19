Max Unpooling Layer
=================================

Unfortunately, as some inputs are being dropped in the pooling layer
(because only the maximum value will be kept), we cannot fully inverse
the max-pooling operation. However, if we have the position of each
maximum value when performing max pooling, then we can simply put the
maximum value back to its original position. After putting them back, we
can set values at other positions to be :math:`0`.

As in the pooling section, we have an input matrix
:math:`X=\left[ {\begin{array}{*{20}c} 1 & 2 & 3 \\4 & 5 & 6 \\7 & 8 & 9 \end{array} } \right]` and the corresponding output
:math:`P=\left[ {\begin{array}{*{20}c} 5 & 6 \\8 & 9 \end{array} } \right]`. Besides these, we will know the indices of the
maximum value in each region, which are :math:`(1,1)`, :math:`(1,2)`,
:math:`(2,1)` and :math:`(2,2)`.

In the unpooling process, we first put the maximum values back to its
position, and fill other positions with :math:`0`. We will get the
output as :math:`\left[ {\begin{array}{*{20}c} 0 & 0 & 0 \\0 & 5 & 6 \\ 0 & 8 & 9 \end{array} } \right]`

.. literalinclude:: ../../../tinyml/layers/unpooling.py
  :language: Python
  :linenos:
