Linear Transformation of Matrices
=================================

**Differentiation of Linear Transformation of Matrices** As suggested
above, we will need to compute the derivatives of the loss functions to
the parameters in neural networks. As we usually represent the data
(such as the input, output, other parameters, etc) in neural networks as
matrices, and the most fundamental transformations in neural networks
are linear, it is essential to understand how to compute the derivatives
of a linear transformation of matrices by using the chain rule.

Since there is no such concept “Layer” in this process, we will use
:math:`X` and :math:`Y` to denote the matrices and :math:`x_{ij}` and
:math:`y_{kl}` to represent entries in matrices without the
superscripts.

Assume that we have :math:`f(Y):\mathbb{R}^{m\times n}\to\mathbb{R}` and
a linear tranformation
:math:`g(X):\mathbb{R}^{p\times n}\to \mathbb{R}^{m\times n}, Y=g(X)=AX+B`,
where :math:`A\in\mathbb{R}^{m\times p}, B\in\mathbb{R}^{m\times n}`. We
can compute the derivatives of :math:`f` with respect to :math:`X` as
the following:

-  We know, at the point :math:`x`, if there are two intermediate
   variables :math:`u=\phi(x)` and :math:`v=\psi(x)` that have partial
   derivatives with respect to :math:`x` defined, then the composited
   function :math:`f(u,v)` has partial derivatives with respect to
   :math:`x` defined and can be computed as
   :math:`\frac{\partial f}{\partial x}=\frac{\partial f}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial x}`.
   In our case, there might be several intermediate variables
   :math:`y_{kl}`, hence we have
   :math:`\frac{\partial f}{\partial x_{ij}}=\sum_{kl}\frac{\partial f}{\partial y_{kl}}\frac{\partial y_{kl}}{\partial x_{ij}}`
   (1).

-  Let :math:`a_{ij}` and :math:`b_{ij}` represent elements in :math:`A`
   and :math:`B`, then :math:`y_{kl}=\sum_{s}a_{ks}x_{sl}+b_{kl}`. Hence
   we have
   :math:`\frac{\partial y_{kl}}{\partial x_{ij}}=\frac{\partial \sum_{s}a_{ks}x_{sl}}{\partial x_{ij}}=\frac{\partial a_{ki}x_{il}}{\partial x_{ij}}=a_{ki}\delta_{lj}`
   (2). Here :math:`\delta_{lj}` is defined as :math:`\delta_{lj}=1`
   when :math:`l=j`, otherwise :math:`\delta_{lj}=0`. Intuitively, we
   know that for the single pair :math:`(x_{ij}, y_{kl})`, there is a
   relation :math:`y_{kl}=a_{ki}x_{il}+b_{kl}`. Hence the derivative of
   :math:`y_{kl}` with respect to :math:`x_{ij}` is :math:`a_{ki}` only
   when :math:`l=j`, otherwise, the derivative will be :math:`0`.

-  Take (2) into (1), we will have
   :math:`\frac{\partial f}{\partial x_{ij}}=\sum_{kl}\frac{\partial f}{\partial y_{kl}}\frac{\partial y_{kl}}{\partial x_{ij}}=\sum_{kl}\frac{\partial f}{\partial y_{kl}}a_{ki}\delta_{lj}=\sum_{k}\frac{\partial f}{\partial y_{kj}}a_{ki}`
   (because only :math:`y_{kj}` will be kept). In this equation, we know
   that :math:`a_{ki}` is the :math:`i`\ th row of :math:`A^T` and
   :math:`\frac{\partial f}{\partial y_{kj}}` is the :math:`(k,j)`
   element in the gradient of :math:`f` with respect to :math:`Y`. In
   summary, this equation tells us that the derivative of :math:`f` with
   respect to :math:`x_{ij}` is the dot product of the :math:`i`\ th row
   of :math:`A^T` and the :math:`j`\ th column of :math:`\nabla_Yf`.

-  Now that we have already known that
   :math:`\frac{\partial f}{\partial x_{ij}}` is the dot product of the
   :math:`i`\ th row of :math:`A^T` and the :math:`j`\ th column of
   :math:`\nabla_Yf`. Then for :math:`\frac{\partial f}{\partial X}`, we
   have

   .. math::

      \frac{\partial f}{\partial X}=\left[ {\begin{array}{*{20}c} 
              \frac{\partial f}{\partial x_{11}} & \cdots & \frac{\partial f}{\partial x_{1n}} \\
              \vdots & \frac{\partial f}{\partial x_{ij}} & \vdots \\
              \frac{\partial f}{\partial x_{p1}} & \cdots & \frac{\partial f}{\partial x_{pn}}
              \end{array} } \right]=A^T\nabla_Y f

   because every element in :math:`\frac{\partial f}{\partial X}` equals
   to a inner product of a row and a column.

-  It is also common that :math:`g(X)` is defined as
   :math:`g(X):\mathbb{R}^{m\times p}\to \mathbb{R}^{m\times n}, Y=g(X)=XC+D`
   where :math:`C\in\mathbb{R}^{p\times n}` and
   :math:`D\in\mathbb{R}^{m\times n}`. In this case, we have
   :math:`f(Y):\mathbb{R}^{m\times n}\to\mathbb{R}` defined as well.
   Then to compute :math:`\frac{\partial f}{\partial X}`, we first
   consider :math:`Y^T` and we have :math:`Y^T=(XC+D)^T=C^TX^T+D^T`.
   Then by the laws we have found above, we immediately know that
   :math:`\nabla_{X^T}f=(C^T)^T\nabla_{Y^T} f=C\nabla_{Y^T} f`.
   Therefore, we have
   :math:`\nabla_X f = (\nabla_{X^T}f)^T=(C\nabla_{Y^T} f)^T=(\nabla_{Y^T} f)^TC^T=(\nabla_{Y} f)C^T`

In summary, if we have two functions, :math:`f(Y)` takes a matrix and
returns a scalar, and a linear tranformation function :math:`g(X)`, then
we can perform the differentiation using the chain rule. More
specifically, we have found two laws:

-  If the linear transformation is defined as :math:`g(X)=AX+B` (we call
   this left multiplcation), then we have
   :math:`\nabla_X f=A^T\nabla_Y f`. (Law 1)

-  If the linear transformation is defined as :math:`g(X)=XC+D` (we call
   this right multiplcation), then we have
   :math:`\nabla_X f=(\nabla_Y f)C^T`. (Law 2)

Note: We should be careful about the independent variable. Here we are
computing the gradient of the function :math:`f` with respect to
:math:`X`, we can also compute the gradient of the function :math:`f`
with respect to :math:`C`. In that case, we should use a different law.

These two laws are the most important and fundamental conclusions we
have in the whole work, and we will find out that the essential
components of a convolutional neural network (fully connected,
convolution, etc) are linear transformations and the loss function is
the :math:`f(Y)` that we have here. As a result, we will show how to
transform those components into a linear transformation and these
transformations will form the mainline of the *Section 3 Approach*.
