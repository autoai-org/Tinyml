from tinyml.net import Sequential
from tinyml.layers import Conv2D, Linear, MaxPool2D, Flatten, ReLu, Dropout

def vgg16():

    model = Sequential([])

    # Conv Block 1
    model.add(Conv2D('b_1_conv2d_1', (3, 224, 224), n_filter=16,
                    w_filter=3, h_filter=3, stride=1, padding=1))
    model.add(ReLu('b_1_relu_1'))
    model.add(Conv2D('b_1_conv2d_2', (16, 224, 224), n_filter=16,
                    w_filter=3, h_filter=3, stride=1, padding=1))
    model.add(ReLu('b_1_relu_2'))
    model.add(MaxPool2D('b_1_pool_1', (16, 224, 224),
                        (2, 2), stride=2, return_index=True))

    # Conv Block 2
    model.add(Conv2D('b_2_conv2d_1', (16, 112, 112), n_filter=32,
                    w_filter=3, h_filter=3, stride=1, padding=1))
    model.add(ReLu('b_2_relu_1'))
    model.add(Conv2D('b_2_conv2d_2', (32, 112, 112), 32, 3, 3, 1, 1))
    model.add(ReLu('b_2_relu_2'))
    model.add(MaxPool2D('b_2_pool_1', (32, 112, 112), (2, 2), 2, return_index=True))

    # Conv Block 3
    model.add(Conv2D('b_3_conv2d_1', (32, 56, 56), 64, 3, 3, 1, 1))
    model.add(ReLu('b_3_relu_1'))
    model.add(Conv2D('b_3_conv2d_2', (64, 56, 56), 64, 3, 3, 1, 1))
    model.add(ReLu('b_3_relu_2'))
    model.add(Conv2D('b_3_conv2d_3', (64, 56, 56), 64, 3, 3, 1, 1))
    model.add(ReLu('b_3_relu_3'))
    model.add(MaxPool2D('b_3_pool_1', (64, 56, 56), (2, 2), 2, return_index=True))

    # Conv Block 4
    model.add(Conv2D('b_4_conv2d_1', (64, 28, 28), 128, 3, 3, 1, 1))
    model.add(ReLu('b_4_relu_1'))
    model.add(Conv2D('b_4_conv2d_2', (128, 28, 28), 128, 3, 3, 1, 1))
    model.add(ReLu('b_4_relu_2'))
    model.add(Conv2D('b_4_conv2d_3', (128, 28, 28), 128, 3, 3, 1, 1))
    model.add(ReLu('b_4_relu_3'))
    model.add(MaxPool2D('b_4_pool_1', (128, 28, 28), (2, 2), 2, return_index=True))

    # Conv Block 5
    model.add(Conv2D('b_5_conv2d_1', (128, 14, 14), 256, 3, 3, 1, 1))
    model.add(ReLu('b_5_relu_1'))
    model.add(Conv2D('b_5_conv2d_2', (256, 14, 14), 256, 3, 3, 1, 1))
    model.add(ReLu('b_5_relu_2'))
    model.add(Conv2D('b_5_conv2d_3', (256, 14, 14), 256, 3, 3, 1, 1))
    model.add(ReLu('b_5_relu_3'))
    model.add(MaxPool2D('b_5_pool_1', (256, 14, 14), (2, 2), 2, return_index=True))

    # Classifier
    model.add(Flatten('flat_1'))
    model.add(Linear('fc_1', 256*7*7, 4096))
    model.add(ReLu('relu_1'))
    model.add(Dropout('drop_1', 0.5))
    model.add(Linear('fc_2', 4096, 4096))
    model.add(ReLu('relu_2'))
    model.add(Dropout('drop_2', 0.5))
    model.add(Linear('fc_3', 4096, 2))
    model.build_params()
    return model