Notations
===================================

In the below sections, we will by default use the
following notations:

-  :math:`x_j^{(i)}` is the :math:`j`\ th input of the :math:`i`\ th
   layer. In fully connected layers, all the inputs form a vector, thus
   we can use :math:`x_j^{(i)}` to denote them. However, in
   convolutional layers, the inputs form a :math:`2`\ d matrix, and in
   that case, we will use :math:`x_{kj}^{(i)}` to denote the individual
   input. The formed vectors (in fully connected layers) or matrices (in
   convolutional layers) will be represented by :math:`X^{(i)}`.

-  :math:`\hat{y}_j^{(i)}` is :math:`j`\ th the output of the
   :math:`i`\ th layer. If the :math:`i`\ th layer is the last layer,
   there might be some corresponding ground truths. In this case, the
   ground truths will be denoted as :math:`y_j^{(i)}`. We will use
   :math:`\hat{Y}^{i}` to denote the vector or matrix that contains all
   the outputs of the :math:`i`\ th layer. Correspondingly, the vector
   or matrix of the ground truth will be denoted by :math:`Y^{(i)}`

-  :math:`\ell` is the value of loss function. In our classification
   model, we use the cross-entropy loss and it is defined as
   :math:`\ell = -\sum_{i}^{n}y_j^{(i)} log(\hat{y}_{j}^{(i)})` where
   :math:`i` is the index of the last layer. In regression models, it
   can be defined as
   :math:`\ell=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2` (mean square
   loss). In the below examples, we will by default use the mean square
   loss as it is easier to compute.

-  :math:`\nabla_{X}f` refers to the derivative of function :math:`f`
   with respect to :math:`X`, i.e.
   :math:`\nabla_{X}f=\frac{\partial f}{\partial X}`. In the following
   sections, we will use :math:`\nabla^{(i)}` to represent the gradient
   of the loss value with respect to the output of the :math:`i`\ th
   layer. Formally, we have
   :math:`\nabla^{(i)}`\ =\ :math:`\frac{\partial\ell}{\partial \hat{Y}^{(i)}}`.
   Since the output of the :math:`i`\ th layer is also the input of the
   :math:`(i+1)`\ th layer, we also have
   :math:`\nabla^{(i)}=\frac{\partial\ell}{\partial X^{(i+1)}}`. In case
   :math:`\nabla^{(i)}` is a matrix, we will use
   :math:`\nabla^{(i)}_{kj}` to denote the :math:`(k,j)` element in the
   matrix as well.

-  There are some other notations we need in the following sections:
   :math:`\mathbb{R}` is for the set of real numbers,
   :math:`\mathbb{R}^{m\times n}` is for an :math:`m\times n` real
   matrix and :math:`\epsilon` is a small enough real number.