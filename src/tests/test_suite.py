import unittest

from .test_trainer import TestTrainer
from .test_serial_hierarchy import TestSerialHierarchy
from .test_preprocessor import TestPreprocessor
from .test_actions import TestActions

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTrainer)
    suite.addTest(TestSerialHierarchy)
    suite.addTest(TestPreprocessor)
    suite.addTest(TestActions)
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())