import unittest

from .test_trainer import TestTrainer
from .test_serial_hierarchy import TestSerialHierarchy
from .test_preprocessor import TestPreprocessor
from .test_actions import TestActions
from .test_n_step_sarsa_semi_gradient import TestSARSAn
from .test_semi_gradient_sarsa import TestSemiGradSARSA
from .test_tiled_environment import TestTiledEnv
from .test_epsilon_greedy_q_estimator import TestEpsilonGreedyQEstimator


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTrainer)
    suite.addTest(TestSerialHierarchy)
    suite.addTest(TestPreprocessor)
    suite.addTest(TestActions)
    suite.addTest(TestSARSAn)
    suite.addTest(TestSemiGradSARSA)
    suite.addTest(TestTiledEnv)
    suite.addTest(TestEpsilonGreedyQEstimator)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
