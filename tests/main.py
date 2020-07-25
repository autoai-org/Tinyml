import unittest

from .pooling_test import TestMaxpool2D

suite = unittest.TestSuite()
suite.addTest(TestMaxpool2D('Test Max Pooling 2D'))

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
