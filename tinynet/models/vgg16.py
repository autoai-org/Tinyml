from tinynet.net import Sequential
from tinynet.layers import Conv2D, Linear, MaxPool2D, Flatten, ReLu
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
model.add(MaxPool2D('b_4_pool_1', (512, 8, 8), (2, 2), 2, return_index=True))

# Conv Block 5
model.add(Conv2D('b_5_conv2d_1', (512, 4, 4), 512, 3, 3, 1, 1))
model.add(ReLu('b_5_relu_1'))
model.add(Conv2D('b_5_conv2d_2', (512, 4, 4), 512, 3, 3, 1, 1))
model.add(ReLu('b_5_relu_2'))
model.add(Conv2D('b_5_conv2d_3', (512, 4, 4), 512, 3, 3, 1, 1))
model.add(ReLu('b_5_relu_3'))
model.add(MaxPool2D('b_5_pool_1', (512, 4, 4), (2, 2), 2, return_index=True))


model.summary()
