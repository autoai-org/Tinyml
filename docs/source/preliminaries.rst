Preliminaries
===================================

**Convolutional Neural Network**: Convolutional Neural Network (CNN) is
a class of neural networks, and has been proven to be effective for most
computer vision tasks. In a CNN architecture for image classification,
there are usually three important components: the convolutional layer,
the pooling layer and the fully connected layer. The first two types are
designed to extract high-level features from images, and the fully
connected layer can be used as a classifier to output the classification
results. In a convolutional neural network, the convolutional and fully
connected layers are equipped with parameters called *weight* and
*bias*, which will be updated during training. In this work, we
implemented these components along with other necessary components, such
as activations (ReLu function), losses (Cross-Entropy Loss), etc.

**Deconvolution**: The two fundamental components of the CNN are
convolutional layers and pooling layers, which works together to
transform images into feature maps. Deconvolutional operation is a
transformation that goes in the opposite direction of a normal
convolution operation, i.e. from the feature maps that we extracted with
convolution operation to something that has the shape of the input to
it. After being introduced, deconvolution has been used in many fields
such as pixel-wise segmentation, generative models, etc. In this work,
we use deconvolution to map the intermediate feature maps back to the
input pixel space. With this approach, we can show the relationship
between the input patterns and the activations in the feature maps.
There are also two components in the approach, called deconvolutional
layers and unpooling layers, and we will explain these two concepts in
more detail in *Section 3 Approach*.

**Stochastic Gradient Descent**: In neural networks, we want to find a
function :math:`\hat{y}=F(x)` such that :math:`y-F(x)` is minimal. The
function that we are looking for is usually non-linear, non-convex and
there is generally no formula for it. As a result, gradient descent
becomes one of the most popular methods to find the local minimum of the
function. The method is based on a fact that the function :math:`f`
decreases fastest along the direction of the negative gradient.
Formally, we can define a function that measures the difference between
:math:`\hat{y}` and :math:`y`, for example :math:`f=y-\hat{y}` and
assume that :math:`a` is the only parameter in :math:`f`. Then if we let
:math:`a_{n+1}=a_n-\epsilon\nabla_af` and we want to find the lowest
value of :math:`f(a)` around the point :math:`a`. Then if
:math:`\epsilon` is small enough, we will have
:math:`f(a_{n+1})\leq f(a_n)` and :math:`f(a_{n+1})` is the smallest
value around a small enough interval of :math:`a`. Considering this, if
we want to find the local minimal of the function :math:`f`, we can
start at a random point :math:`a_0`, and follows the negative direction
of the gradient. With this approach, we will have a sequence
:math:`a_1, a_2,\cdots a_n` that satisfies
:math:`a_{n+1}=a_n-\epsilon\nabla_af`. Then the output of the function
:math:`f` will satisfy the rule that
:math:`f(a_n)\leq f(a_{n-1})\leq\cdots \leq f(a_{0})`. By doing so, we
could find an approximate value :math:`a_n` such that :math:`f(a_n)` is
the local minimal.

**Backpropagation**: In the process of gradient descent, we found that
we need to compute the gradient of our function :math:`f` in every step.
Backpropagation, as an application of the chain rule, is an efficient
algorithm for calculating the gradient in deep neural networks. In
short, it first computes the gradient of the loss function to the weight
of the last layer in a neural network, and passes the gradient of the
loss function to the input of the layer to previous layers. There are
two bases for the algorithm:

-  In the :math:`i`\ th layer, we can receive the gradient of the loss
   :math:`\ell` with respects to the output of :math:`i`\ th layer, i.e.
   :math:`\frac{\partial \ell}{\partial \hat{y}^{(i)}}` is known to us.

-  Since the output of :math:`(i-1)`\ th layer is the input of the
   :math:`i`\ th layer, we have
   :math:`\frac{\partial \ell}{\partial x^{(i)}}=\frac{\partial \ell}{\partial \hat{y}^{i-1}}`

Having these two bases, we could compute the gradient of the loss
:math:`\ell` with respect to the weight and input of every layer by
applying chain rules. For example,
:math:`\frac{\partial \ell}{\partial w^{(i)}}=\frac{\partial \ell}{\partial \hat{y}^{(i)}}\frac{\partial \hat{y}^{(i)}}{\partial w^{(i)}}`
where we only need to know how to compute :math:`\hat{y}^{(i)}` with
:math:`w^{(i)}`. With these, we could efficiently compute the loss value
with respect to every parameter in the neural network and update them
with the SGD method.

**Numerical Differentiation** Besides manually working out the
derivatives, we can also estimate the derivatives with numerical
approximation. numerical differentiation is an algorithm for estimating
the derivative of a mathematical function using the values of the
function.

The simplest method, also known as Newton’s differentiation quotient is
by using the finite difference approximations. More specifically, if we
have a function :math:`f(x)` and we want to compute the derivative of
:math:`f`, we can approximate it by computing the slope of a nearby
secant line through the points :math:`(x, f(x))` and
:math:`(x+\epsilon, f(x+\epsilon))`. The slope of this secant line will
be :math:`\frac{f(x+\epsilon)-f(x)}{\epsilon}`, and the derivative is
the tangent line at the point :math:`x`. As :math:`\epsilon` approaches
:math:`0`, the slope of the secant line approaches the slope of the
tangent line. Therefore, the true derivative of :math:`f` at the point
:math:`x` can be defined as
:math:`f'(x)=\lim_{\epsilon\to0}\frac{f(x+\epsilon)-f(x)}{\epsilon}`.
Then by this nature, we could manually choose a small enough
:math:`\epsilon`, and to approximately approach the tangent line, i.e.
the derivative of the function :math:`f` at :math:`x`.

As we know the point :math:`x+\epsilon` is at the right of :math:`x`,
the form :math:`\frac{f(x+\epsilon)-f(x)}{\epsilon}` is called
right-sided form. Besides this form, we can also approach the tangent
line from the left side and right side (the two-sided form) at the same
time. To do so, we compute the slope of a nearby secant line through the
points :math:`(x-\epsilon, f(x-\epsilon))` and
:math:`(x+\epsilon, f(x+\epsilon))` by
:math:`\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}`. This form is a
more accurate approximation to the tangent line than the one-sided form
and therefore we will use the two-sided form in the following sections.

In order to verify that we are working out the derivatives correctly, we
will involve the numerical differentiation as a way of cross-validation,
and increase our confidence in the correctness of induction. In *3.1.1
Fully Connected*, we will show how to perform the numerical
differentiation in a concrete and simple example.
