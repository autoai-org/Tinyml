import numpy as np
from PIL import Image

from tinyml.layers import Conv2D, Deconv2D, MaxUnpool2D, ReLu
from tinyml.net import Sequential

conv2deconv_indices = {
    0: 23,
    2: 21,
    5: 18,
    7: 16,
    10: 13,
    12: 11,
    14: 9,
    17: 6,
    19: 4,
    21: 2
}

pooling2unpooling_indices = {4: 19, 9: 14, 16: 7, 23: 0}

unpooling2pooling_indices = {19: 4, 14: 9, 7: 16, 0: 23}


def inverse_vgg():
    model = Sequential([])

    # Deconv Block 1 <-> Conv Block 4
    model.add(MaxUnpool2D('db_1_pool_1', (64, 14, 14), (2, 2), 2))
    model.add(ReLu('db_1_relu_1'))
    model.add(Deconv2D('db_1_deconv_1', (64, 28, 28), 64, 3, 3, 1, 1, 1))
    model.add(ReLu('db_1_relu_2'))
    model.add(Deconv2D('db_1_deconv_2', (64, 28, 28), 64, 3, 3, 1, 1, 1))
    model.add(ReLu('db_1_relu_3'))
    model.add(Deconv2D('db_1_deconv_3', (64, 28, 28), 32, 3, 3, 1, 1, 1))

    # Deconv Block 2 <-> Conv Block 3
    model.add(MaxUnpool2D('db_2_pool_1', (32, 28, 28), (2, 2), 2))
    model.add(ReLu('db_2_relu_1'))
    model.add(Deconv2D('db_2_deconv_1', (32, 56, 56), 32, 3, 3, 1, 1, 1))
    model.add(ReLu('db_2_relu_2'))
    model.add(Deconv2D('db_2_deconv_2', (32, 56, 56), 32, 3, 3, 1, 1, 1))
    model.add(ReLu('db_2_relu_3'))
    model.add(Deconv2D('db_2_deconv_3', (32, 56, 56), 16, 3, 3, 1, 1, 1))

    # Deconv Block 3 <-> Conv Block 2
    model.add(MaxUnpool2D('db_3_pool_1', (16, 56, 56), (2, 2), 2))
    model.add(ReLu('db_3_relu_1'))
    model.add(Deconv2D('db_3_deconv_1', (16, 112, 112), 16, 3, 3, 1, 1, 1))
    model.add(ReLu('db_3_relu_2'))
    model.add(Deconv2D('db_3_deconv_2', (16, 112, 112), 8, 3, 3, 1, 1, 1))

    # Deconv Block 4 <-> Conv Block 1
    model.add(MaxUnpool2D('db_3_pool_1', (8, 112, 112), (2, 2), 2))
    model.add(ReLu('db_4_relu_1'))
    model.add(Deconv2D('db_4_deconv_1', (8, 224, 224), 8, 3, 3, 1, 1, 1))
    model.add(ReLu('db_4_relu_2'))
    model.add(Deconv2D('db_4_deconv_2', (8, 224, 224), 3, 3, 3, 1, 1, 1))
    return model


def load_weight(inv_vgg, vgg):
    for idx, layer in enumerate(vgg.layers):
        if isinstance(layer, Conv2D):
            inv_vgg.layers[
                conv2deconv_indices[idx]].weight.tensor = layer.weight.tensor
            # No idea why we ignore bias, but that's what [https://github.com/huybery/VisualizingCNN] did.
            # inv_vgg.layers[conv2deconv_indices[idx]].bias.tensor = layer.bias.tensor
    return inv_vgg


def forward(x, inv_vgg, layer_id, pool_indices):
    if layer_id in conv2deconv_indices:
        start_idx = conv2deconv_indices[layer_id]
    else:
        raise ValueError("Not a convolutional feature maps to start")
    for idx in range(start_idx, len(inv_vgg.layers)):
        if isinstance(inv_vgg.layers[idx], MaxUnpool2D):
            x = inv_vgg.layers[idx](
                x, pool_indices[unpooling2pooling_indices[idx]])
        else:
            x = inv_vgg.layers[idx](x)
    return x
