Convolutional Layer
===================

**Definition** In fully connected layers, the input is always a
one-dimensional vector. However, images are usually stored as a
multi-dimensional matrix and have implicit spatial structures. For
example, the eyes are always on top of the nose, etc. These properties
are not well expressed using a fully connected layer. Hence we use the
convolutional layers to preserve these properties. For example, assume
we have a single channel input matrix :math:`A_{3\times 3}` and single
filter matrix :math:`B_{2\times 2}`, and

.. math::

   A=\left[ {\begin{array}{*{20}c} 
           a_{11} & a_{12} & a_{13} \\
           a_{21} & a_{22} & a_{23} \\
           a_{31} & a_{32} & a_{33}   
           \end{array} } \right], B=\left[ {\begin{array}{*{20}c}
           b_{11} & b_{12}    \\
           b_{21} & b_{22}   
       \end{array} } \right]

Then, we slide the filter :math:`B` with a unit stride, i.e. move one
column or one row at a time. If we use :math:`a_{ij}` and :math:`b_{ij}`
to denote the element in :math:`A` and :math:`B` at the :math:`(i,j)`
location. Then we can obtain the output of the convolutional layer with
the following steps:

-  At the beginning, the :math:`2\times 2` filter is placed at the upper
   left corner. At this time, we perform the dot product and we will
   have
   :math:`y_{11}=a_{11}b_{11}+a_{12}b_{12}+a_{21}b_{21}+a_{22}b_{22}`.

-  Then we slide the filter across the width for a unit stride, i.e. we
   move the slide to the upper right corner. At this time, we perform
   the dot product and we will have
   :math:`y_{12}=a_{12}b_{11}+a_{13}b_{12}+a_{22}b_{21}+a_{23}b_{22}`.

-  Then we found that there is no more values on the right side, so we
   start to slide the filter on the next row. At this time, we start at
   the bottom left corner and we can obtain that
   :math:`y_{21}=a_{21}b_{11}+a_{22}b_{12}+a_{31}b_{21}+a_{32}b_{22}`.

-  Then we again slide the filter to the right side, i.e. the bottom
   right corner and we obtain that
   :math:`y_{22}=a_{22}b_{11}+a_{23}b_{12}+a_{32}b_{21}+a_{33}b_{22}`.

After these steps, we found that we get four outputs, and we can obtain
the final output if we place the output values to corresponding
locations, i.e. the value we computed at the upper left corner is placed
at the upper left corner and so on so forth. Hence, in this example, we
get a :math:`(1,2,2)` output matrix

.. math:: 

   C=\left[ {\begin{array}{*{20}c}
         y_{11} & y_{12}    \\
         y_{21} & y_{22}   
      \end{array} } \right]

More generally, we can formalize the input, the filters and the output
of a convolutional layer as below:

-  The input is a :math:`(N, C, H, W)` tensor, where :math:`N` is the
   number of the input matrix in a single batch, :math:`C` is the
   channel of the matrix, :math:`H` and :math:`W` are the height and
   width of the input matrix. For example, :math:`10` coloured images
   with the size :math:`(224, 224)` can be represented as a
   :math:`(10, 3, 224, 224)` tensor.

-  The filters is a :math:`(K, C, H_f, W_f)` tensor, where :math:`K` is
   the number of filters, :math:`C` is the channel of the filters and it
   will always be identical to the channel of the input matrix.
   :math:`H_f` and :math:`W_f` are the height and width of the filters.

-  The output is a :math:`(N, K, H_{out}, W_{out})` tensor.

-  The stride that we use to slide the filters are denoted as :math:`S`.

With these notations, we can compute the output of a convolution layer
with seven loops.

Though the convolution operation can be computed by the above algorithm,
we can still use matrix multiplication to perform such computation as
suggested by `A guide to convolution arithmetic for deep
learning <https://arxiv.org/pdf/1603.07285.pdf>`_. The benefits of using matrix multiplication are two-fold:

-  We have already gotten two laws for computing the differentiation of
   linear transformation. If we can define the convolution operation as
   :math:`g(x)=AX+B` (i.e. matrix multiplication), we could easily reuse
   the two laws and get the derivative of the loss value with respect to
   the filter and input.

-  We are about to study how to compute the forward pass of Deconv
   operation and that operation can be easily defined with the matrix
   multiplication form of the convolution operation. We will see this in
   *Section 3.2.1 Deconv*.

Hence, in the below computation of the forward pass and backward pass of
the convolution operation, we will show how to convert it into a matrix
multiplication.

**Forward Pass** Recall that the input :math:`X`, the filter :math:`W`
and the expected output :math:`Y` we have in the above example.

.. math::

   X=\left[ {\begin{array}{*{20}c} 
       a_{11} & a_{12} & a_{13} \\
       a_{21} & a_{22} & a_{23} \\
       a_{31} & a_{32} & a_{33}   
       \end{array} } \right], W=\left[ {\begin{array}{*{20}c}
       b_{11} & b_{12}    \\
       b_{21} & b_{22}   
   \end{array} } \right], Y=\left[ {\begin{array}{*{20}c}
       y_{11} & y_{12}    \\
       y_{21} & y_{22}   
   \end{array} } \right]

If we unroll the input and the output into vectors from left to right,
top to bottom, we can also represent the filters as a sparse matrix
where the non-zero elements are the elements in the filters. For
example, We can unroll the input and output in our case as
:math:`X^*_{9\times 1}=[a_{11},a_{12},a_{13},a_{21},a_{22},a_{23},a_{31},a_{32},a_{33}]^T`
and :math:`Y^*_{4\times 1}=[y_{11},y_{12},y_{21} ,y_{22}]^T`. Then we
will want a :math:`4\times 9` matrix :math:`W^*` such that
:math:`Y^* = W^*X^*`. From the direct computation of convolutional
layers, we can transform the original filters :math:`W` into

.. math::

   W^*=\left[ {\begin{array}{*{20}c} 
           b_{11} & b_{12} & 0 & b_{21} & b_{22} & 0 & 0 & 0 & 0 \\
           0 & b_{11} & b_{12} & 0 & b_{21} & b_{22} & 0 & 0 & 0 \\
           0 & 0 & 0 & b_{11} & b_{12} & 0 & b_{21} & b_{22} & 0 \\
           0 & 0 & 0 & 0 & b_{11} & b_{12} & 0 & b_{21} & b_{22}
       \end{array} } \right]

Then we can verify that

.. math::

   W^*X^*=\left[ {\begin{array}{*{20}c} 
       b_{11} & b_{12} & 0 & b_{21} & b_{22} & 0 & 0 & 0 & 0 \\
       0 & b_{11} & b_{12} & 0 & b_{21} & b_{22} & 0 & 0 & 0 \\
       0 & 0 & 0 & b_{11} & b_{12} & 0 & b_{21} & b_{22} & 0 \\
       0 & 0 & 0 & 0 & b_{11} & b_{12} & 0 & b_{21} & b_{22}
   \end{array} } \right] \times [a_{11},a_{12},a_{13}\cdots, a_{33}]^T = Y^*

**Backward Pass** From the forward pass, we converted the filters into a
sparse matrix :math:`W^*`, and we found that the convolution is also a
linear operation, i.e. :math:`Y^*=W^*X^*`. Similar to the backward pass
of fully connected layers, we can directly compute the gradient of the
loss value with respect to the weight matrix as
:math:`\frac{\partial\ell}{\partial W^*}=\nabla^{(i)} (X^*)^T` (Using
the Law 2). Hence, we can update the weight matrix in convolutional
layers as :math:`(W^*)^{new}=(W^*)^{old}-\epsilon \nabla^{(i)}(X^*)^T`
and it is the same as in fully connected layers.

Besides, we will need to compute the gradient that we want to pass to
previous layers, i.e. the gradient of the loss value with respect to the
input matrix. We will have
:math:`\frac{\partial\ell}{\partial X^*}=\frac{\partial\ell}{\partial Y^*}\frac{\partial Y^*}{\partial X^*}=(W^*)^T\nabla^{(i)}`
(Using the Law 1).

Here we will show how the unrolling process works for convolution
operation and how to perform the forward pass in the direct and matrix
multiplication methods. Since the backward pass is identical to fully
connected layers, we will not compute the backward pass in this example.

We assume that we have an input :math:`X`, the filter :math:`W` as

.. math::

   X=\left[ {\begin{array}{*{20}c} 
       1 & 2 & 3 \\
       4 & 5 & 6 \\
       7 & 8 & 9   
       \end{array} } \right], W=\left[ {\begin{array}{*{20}c}
       1 & 2 \\
       3 & 4
   \end{array} } \right]

*Direct Computation* To compute the output directly, we will have to
slide the filter :math:`W` from left to right, from top to bottom. We
will have :math:`1*1+2*2+4* 3+5*4=37`, :math:`2*1+3*2+5*3+6*4=47`,
:math:`4*1+5*2+7*3+8*4=67`, :math:`5*1+6*2+8*3+9*4=77` in the upper
left, upper right, bottom left and bottom right corners. Then, we can
get the output as

.. math::

   Y=\left[ {\begin{array}{*{20}c}
       37 & 47 \\
       67 & 77
   \end{array} } \right]

*Matrix Multiplcation* With matrix multiplication approach, we need to
unroll the filter and input matrices into

.. math::

   W^*=\left[ {\begin{array}{*{20}c} 
       1 & 2 & 0 & 3 & 4 & 0 & 0 & 0 & 0 \\
       0 & 1 & 2 & 0 & 3 & 4 & 0 & 0 & 0 \\
       0 & 0 & 0 & 1 & 2 & 0 & 3 & 4 & 0 \\
       0 & 0 & 0 & 0 & 1 & 2 & 0 & 3 & 4
   \end{array} } \right], X^*=\left[ {\begin{array}{*{20}c} 
       1 & 2 & \cdots & 9 
       \end{array} } \right]^T

Then we can compute the output directly by
:math:`Y^*=W^*X^*=[37,47,67,77]^T`. By reshaping :math:`Y^*`, we can
easily obtain the desired output matrix :math:`Y_{2\times 2}`.

Since the backward process will be identical to what we did in *Section
3.1.1 Fully Connected* if we perform the forward pass in a matrix
multiplication way, we will omit the examples here.