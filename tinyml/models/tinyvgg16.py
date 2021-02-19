from tinynet.net import Sequential
from tinynet.layers import Conv2D, Linear, MaxPool2D, Flatten, ReLu, Dropout

'''
Tiny VGG 16 is a vgg-like architecture, 
to tackle the tinynet challenge.
TensorFlow version: https://github.com/pat-coady/tiny_imagenet/blob/master/src/vgg_16.py
'''

def tinyvgg16():

    model = Sequential([])

    # Conv Block 1
    model.add(Conv2D('b_1_conv2d_1', (3, 64, 64), n_filter=64,
                    w_filter=3, h_filter=3, stride=1, padding=1))
    model.add(ReLu('b_1_relu_1'))
    model.add(Conv2D('b_1_conv2d_2', (64, 64, 64), n_filter=64,
                    w_filter=3, h_filter=3, stride=1, padding=1))
    model.add(ReLu('b_1_relu_2'))
    model.add(MaxPool2D('b_1_pool_1', (64, 64, 64),
                        (2, 2), stride=2, return_index=True))

    # Conv Block 2
    model.add(Conv2D('b_2_conv2d_1', (64, 32, 32), n_filter=128,
                    w_filter=3, h_filter=3, stride=1, padding=1))
    model.add(ReLu('b_2_relu_1'))
    model.add(Conv2D('b_2_conv2d_2', (128, 32, 32), 128, 3, 3, 1, 1))
    model.add(ReLu('b_2_relu_2'))
    model.add(MaxPool2D('b_2_pool_1', (128, 32, 32), (2, 2), 2, return_index=True))

    # Conv Block 3
    model.add(Conv2D('b_3_conv2d_1', (128, 16, 16), 256, 3, 3, 1, 1))
    model.add(ReLu('b_3_relu_1'))
    model.add(Conv2D('b_3_conv2d_2', (256, 16, 16), 256, 3, 3, 1, 1))
    model.add(ReLu('b_3_relu_2'))
    model.add(Conv2D('b_3_conv2d_3', (256, 16, 16), 256, 3, 3, 1, 1))
    model.add(ReLu('b_3_relu_3'))
    model.add(MaxPool2D('b_3_pool_1', (256, 16, 16), (2, 2), 2, return_index=True))

    # Conv Block 4
    model.add(Conv2D('b_4_conv2d_1', (256, 8, 8), 512, 3, 3, 1, 1))
    model.add(ReLu('b_4_relu_1'))
    model.add(Conv2D('b_4_conv2d_2', (512, 8, 8), 512, 3, 3, 1, 1))
    model.add(ReLu('b_4_relu_2'))
    model.add(Conv2D('b_4_conv2d_3', (512, 8, 8), 512, 3, 3, 1, 1))
    model.add(ReLu('b_4_relu_3'))

    # Classifier
    model.add(Flatten('flat_1'))
    model.add(Linear('fc_1', 512*8*8, 4096))
    model.add(ReLu('relu_1'))
    model.add(Dropout('drop_1', 0.5))
    model.add(Linear('fc_2', 4096, 4096))
    model.add(ReLu('relu_2'))
    model.add(Dropout('drop_2', 0.5))
    model.add(Linear('fc_3', 4096, 200))

    model.build_params()
    return model