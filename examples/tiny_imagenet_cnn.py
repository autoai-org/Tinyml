import tinynet.dataloaders.imagenet as tinyimagenet

import tinynet

# Higher verbose level = more detailed logging
# tinynet.utilities.logger.VERBOSE = 1

print('loading data...')

x_train, y_train, x_test, y_test = tinyimagenet.load_tiny_imagenet('./dataset/tinyimagenet.pkl')

print(x_train.shape)