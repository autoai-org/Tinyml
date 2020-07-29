import unittest

from tests.conv_test import TestConv2D, TestConv2D_multiple_channel
from tests.pooling_test import TestMaxpool2D, TestMaxpool2DPyTorch
from tests.relu_test import TestRelu
from tests.unpooling_test import TestMaxUnpool2D, TestUnpool2DwithTorch


def convolution():
    suite = unittest.TestSuite()
    suite.addTest(TestConv2D('test_convolution'))
    suite.addTest(TestConv2D_multiple_channel('test_convolution'))
    return suite


def pooling():
    suite = unittest.TestSuite()
    suite.addTest(TestMaxpool2D('test_single_channel_forward'))
    suite.addTest(TestMaxpool2DPyTorch('test'))
    return suite


def relu():
    suite = unittest.TestSuite()
    suite.addTest(TestRelu('test'))
    return suite


def unpooling():
    suite = unittest.TestSuite()
    suite.addTest(TestMaxUnpool2D('test_single_channel_forward'))
    suite.addTest(TestUnpool2DwithTorch('test'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    print('-- conv2d --')
    runner.run(convolution())
    print('-- max pooling --')
    runner.run(unpooling())
    print('-- relu --')
    runner.run(relu())
    print('-- max unpooling --')
    runner.run(unpooling())
