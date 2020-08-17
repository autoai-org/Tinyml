Overview
===================================

I wrote Tinynet for learning purpose, and I found it is useful and interesting to figure out what is a deep learning framework doing under their apis. I hope this project can be useful for other students to understand how deep learning works.

In Tinynet, we focus on three main tasks: construct the
neural networks, perform the training, evaluating and visualizing
processes and export the trained weight to the persistent storage (i.e.
the hard disk). We have made the following modules to achieve these
goals:

-  Core. In this module, we implement a base class for all the
   parameters that need to be updated during training. A parameter
   includes two components: the tensor that saves the actual data, and
   the gradient that saves the derivatives of the loss with respect to
   the parameter for updating.

-  Layers. We implement all the needed layers in this module, including
   the fully connected layer, the convolutional layer, ReLu and Dropout
   layer, etc. All these layers are Python classes that are extended
   from a base class, which requires the subclasses to implement a
   *forward* and a *backward* function.

-  Losses. We implement the needed cross-entropy loss in this module.
   The loss function is implemented as a Python function that has two
   inputs, *predicted* and *ground truth*. Then the function needs to
   return two values, the *loss value*, which measures the distance
   between the ground truth and the predicted output, and the
   *gradient*, which calculates the derivatives of loss value with
   respect to the predicted output.

-  Net. Net is a class that stacks several different layers, and
   provides three functions: *forward*, *backward* and *update*. The
   forward function will compute the output of the forward pass from the
   beginning of the stacked layers, while the backward function will
   first reverse those layers and then compute the backward pass from
   the end of the given layers. The update function simply updates all
   those parameters in a neural network at once.

-  Optimizer. The SGD optimizer is implemented in this module. The
   optimizer receives a parameter from the *Core* module, computes the
   next value by :math:`new=old-\epsilon\nabla` where :math:`\epsilon`
   is the preset learning rate, and :math:`\nabla` is the computed
   derivative of the loss with respect to the parameter.

-  Learner. We perform the actual training process inside the *Learner*
   module. A learner receives a user-defined neural network
   architecture, a training dataset, an optimizer and some other
   hyperparameters such as batch size. Then the learner will read the
   training dataset batch by batch, and in each batch, the learner will
   call the forward function of the given neural network architecture on
   the batch, compute the loss value and then perform the backward pass.
   After the backward pass in each batch, the learner will update all
   the parameters in the network.
