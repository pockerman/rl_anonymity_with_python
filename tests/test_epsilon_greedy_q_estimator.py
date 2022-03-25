import unittest
import pytest

from src.algorithms.epsilon_greedy_q_estimator import EpsilonGreedyQEstimator, EpsilonGreedyQEstimatorConfig
from src.exceptions.exceptions import InvalidParamValue


class TestEpsilonGreedyQEstimator(unittest.TestCase):


    def test_constructor(self):
        eps_q_estimator_config = EpsilonGreedyQEstimatorConfig()
        eps_q_estimator = EpsilonGreedyQEstimator(eps_q_estimator_config)
        self.assertIsNotNone(eps_q_estimator)

    def test_on_state(self):
        eps_q_estimator_config = EpsilonGreedyQEstimatorConfig()
        eps_q_estimator = EpsilonGreedyQEstimator(eps_q_estimator_config)

    def test_q_hat_value_raise_InvalidParamValue(self):
        eps_q_estimator_config = EpsilonGreedyQEstimatorConfig()
        eps_q_estimator = EpsilonGreedyQEstimator(eps_q_estimator_config)

        with pytest.raises(InvalidParamValue) as e:
            eps_q_estimator.q_hat_value(None)


if __name__ == '__main__':
    unittest.main()
