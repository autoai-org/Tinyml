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

Mean Square Loss
================

The mean square error is defined as
:math:`l = \frac{1}{n}\sum (y_i-\hat{y}^i)^2`. Since this is the last
derivative we need to compute, we will only need to compute
:math:`\frac{\partial l}{\partial y_i}`. Let
:math:`g(y_i)=y_i-\hat{y_i}`, then
:math:`\frac{\partial g}{\partial y_i}=1`.

.. math:: \frac{\partial l}{\partial y_i}=\frac{\partial l}{\partial g}\times \frac{\partial g}{{\partial y_i}}=\frac{2}{n}(y_i-\hat{y_i})

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

Dropout Layer
=============

In deep neural networks, we may encounter over fitting when our network
is complex and with many parameters. In
:raw-latex:`\cite{JMLR:v15:srivastava14a}`, N.Srivastava et al proposed
a simple technique named *Dropout* that could prevent overfitting. It
refers to dropping out some neurons in a neural network randomly. The
mechanism is equivalent to training different neural networks with
different architecture in every batch.

The parameters for this layer is a preset probability :math:`p_0`. It
indicates the probability of dropping a neuron. For example, if
:math:`p_0=0.5`, then it means that every neuron in this layer has a
:math:`0.5` chance of being dropped. With the given probability, we can
define the dropout layer to be a function :math:`f(x_i)` such that

.. math::

   y_i=
     \begin{cases}
       $0$ & \text{$r_i<p$}  \\
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
