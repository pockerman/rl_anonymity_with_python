import unittest

from .test_trainer import TestTrainer
from .test_serial_hierarchy import TestSerialHierarchy
from .test_preprocessor import TestPreprocessor
from .test_actions import TestActions
from .test_sarsa_semi_gradient import TestSARSAn


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTrainer)
    suite.addTest(TestSerialHierarchy)
    suite.addTest(TestPreprocessor)
    suite.addTest(TestActions)
    suite.addTest(TestSARSAn)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
