Dropout Layer
=============

In deep neural networks, we may encounter over fitting when our network
is complex and with many parameters. In `Dropout: A Simple Way to Prevent Neural
Networks from Overfitting <http://jmlr.org/papers/v15/srivastava14a.html>`_,
N.Srivastava et al proposed
a simple technique named *Dropout* that could prevent overfitting. It
refers to dropping out some neurons in a neural network randomly. The
mechanism is equivalent to training different neural networks with
different architecture in every batch.

The parameters for this layer is a preset probability :math:`p_0`. It
indicates the probability of dropping a neuron. For example, if
:math:`p_0=0.5`, then it means that every neuron in this layer has a
:math:`0.5` chance of being dropped. With the given probability, we can
define the dropout layer to be a function :math:`y_i=f(x_i)` such that

.. math::

   y_i=
     \begin{cases}
       0 & \text{$r_i<p$}  \\
       x_i & \text{$r_i\geq p$} \\
     \end{cases}

, where :math:`r_i` is randomly generated. However, if we use this
function, the expectations of the output of dropout layer will be scaled
to :math:`p_0`. For example, if the original output is :math:`1` and
:math:`p_0=0.5`, the output will become :math:`0.5`. This is
unsatifactory because when we are testing the neural networks, we do not
want the output to be scaled. Thus, in practice we define the function
to be

.. math::

   y_i=
     \begin{cases}
       0 & \text{$r_i<p$}  \\
       x_i/p & \text{$r_i\geq p$} \\
     \end{cases}

Then the backward computation becomes straightforward:

.. math::

   \frac{\partial l}{\partial x_i}=
     \begin{cases}
       0 \times \frac{\partial l}{\partial y_i}=0 & \text{$r_i<p$}  \\
       \frac{\partial l }{\partial y_i}\times\frac{\partial y_i}{\partial x_i}=\frac{1}{p}\frac{\partial l}{\partial y_i} & \text{$r_i\geq p$} \\
     \end{cases}

The implementation of dropout layer in tinyml is as below:

.. literalinclude:: ../../../tinyml/layers/dropout.py
  :language: Python
  :linenos:
