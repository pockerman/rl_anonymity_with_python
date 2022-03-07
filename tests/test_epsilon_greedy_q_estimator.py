import unittest

from src.algorithms.epsilon_greedy_q_estimator import EpsilonGreedyQEstimator, EpsilonGreedyQEstimatorConfig


class TestEpsilonGreedyQEstimator(unittest.TestCase):


    def test_constructor(self):
        eps_q_estimator_config = EpsilonGreedyQEstimatorConfig()
        eps_q_estimator = EpsilonGreedyQEstimator(eps_q_estimator_config)
        self.assertIsNotNone(eps_q_estimator)

    def test_on_state(self):
        eps_q_estimator_config = EpsilonGreedyQEstimatorConfig()
        eps_q_estimator = EpsilonGreedyQEstimator(eps_q_estimator_config)








if __name__ == '__main__':
    unittest.main()