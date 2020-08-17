Transposed Convolutional Layer
=================================

**Definition** In *Section 3.1.2 Convolution*, we unrolled the filter
from a :math:`2\times 2` matrix into a :math:`4\times 9` matrix, so that
we can perform the convolution by matrix multiplication. After the
convolution operation, the input data changes from a :math:`3\times 3`
matrix to a :math:`2\times 2` matrix. The deconv operation is defined as
the inverse of the convolution operation, i.e. change the input data
from a :math:`2\times 2` matrix into an output matrix with the shape
:math:`3\times 3` in our example. The deconv operation does not
guarantee that we will have the same values in the output as the
original matrix. Below we will show how it is computed in the forward
pass.

**Forward Pass** When computing the forward pass of deconv operation, we
can simply transpose the unrolled filters matrix, for example, it will
be a :math:`4\times 9` matrix in our case. After the transpose, we can
define the deconv operation as :math:`X=(W^*)^T Y`, i.e. we use the
transposed, and unrolled filter matrix to multiply the output of the
convolution operation.

We assume that we have an input :math:`Y`\ ( exactly the same with the
output of the convolution operation in our previous example, hence we
will use :math:`Y` as the notation for this input) and the same filter
:math:`W` as

.. math::

   Y=\left[ {\begin{array}{*{20}c} 
       37 & 47 \\
       67 & 77 
       \end{array} } \right], W=\left[ {\begin{array}{*{20}c}
       1 & 2 \\
       3 & 4
   \end{array} } \right]

Then we want to get a :math:`3\times 3` matrix as the output of the
deconv operation. Recall that we unrolled the filter into the matrix as

.. math::

   W^*=\left[ {\begin{array}{*{20}c} 
       1 & 2 & 0 & 3 & 4 & 0 & 0 & 0 & 0 \\
       0 & 1 & 2 & 0 & 3 & 4 & 0 & 0 & 0 \\
       0 & 0 & 0 & 1 & 2 & 0 & 3 & 4 & 0 \\
       0 & 0 & 0 & 0 & 1 & 2 & 0 & 3 & 4
   \end{array} } \right]

We can compute the desired matrix by performing transpose on the filter
matrix first, and then multiply it with our input. We will have

.. math:: X=(W^*)^TY_{4\times 1}=[37, 121, 94, 178, 500, 342, 201, 499, 308]^T

Then we can reshape it back into a :math:`3\times 3` matrix as
:math:`X_{3\times 3}=\left[ {\begin{array}{*{20}c}37 & 121 & 94 \\178 & 500 & 342 \\201 & 499 & 308 \end{array} } \right]`

As we see in this example, the deconv operation does not guarantee that
we will have the same input of convolution operation, but just guarantee
we will have a matrix with the same shape as the input of convolution
operation. Since the entries may exceed the maximum light intensity,
i.e. :math:`255`, when we are visualizing the deconv result, we will
need to renormalize every entry into the range of :math:`[0,255]`.

.. literalinclude:: ../../../tinynet/layers/deconvolution.py
  :language: Python
  :linenos:
