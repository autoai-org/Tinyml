========================
 Theoretical Induction
========================

In the following induction, we will by default define the following
symbols:

-  :math:`w_i` and :math:`b_i`: the weight and bias of the :math:`i`-th
   layer in a neural network.

-  :math:`x_i`: the input to the :math:`i`-th layer in a neural network.

-  :math:`y_i`: the output of the :math:`i`-th layer. On the contrary,
   if there is corresponding ground truth label, it is notated as
   :math:`\hat{y}_i`.

-  :math:`l`: the value of loss function. It can be defined as
   :math:`l=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2` (Mean square
   error), :math:`l=-\sum_{i}^n\hat{y_i}log(y_i)` (cross entropy loss)
   or any other form.

There are two fundamental observations in back propagation:

-  In the :math:`i`-th layer, we always know the gradient of the loss
   with respects to the output of :math:`i`-th layer. That means, in
   :math:`i`-th layer, :math:`\frac{\partial l}{\partial y_i}` is given.

-  Since we know that the output of :math:`(i-1)`-th layer is the input
   of :math:`i`-th layer, when performing backward pass, we have
   :math:`\frac{\partial{l}}{\partial{x_i}}=\frac{\partial{l}}{\partial{y_{i-1}}}`.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   induction/linear-transformation
   induction/linear
   induction/relu
   induction/dropout
   induction/softmax
   induction/pooling
   induction/convolution
   induction/mse
   induction/cross-entropy