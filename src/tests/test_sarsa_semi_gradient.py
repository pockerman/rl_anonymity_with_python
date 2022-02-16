import unittest
import pytest
from src.algorithms.sarsa_semi_gradient import SARSAnConfig, SARSAn
from src.spaces.tiled_environment import TiledEnv, TiledEnvConfig
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption
from src.exceptions.exceptions import InvalidParamValue

class TestSARSAn(unittest.TestCase):

    def test_actions_before_training_throws_invalid_environment(self):

        config = SARSAnConfig()
        agent = SARSAn(sarsa_config=config)
        with pytest.raises(ValueError) as e:
            agent.actions_before_training(env=None)

            self.assertEqual("The given environment does not "
                             "satisfy the IS_TILED_ENV_CONSTRAINT constraint", str(e))

    def test_actions_before_training_throws_invalid_policy(self):
        env_config = TiledEnvConfig()
        env_config.max_size = 4096
        env_config.num_tilings = 5
        env_config.tiling_dim = 6
        env_config.env = None
        env_config.column_scales = {"col1": [0.0, 1.0]}

        env = TiledEnv(env_config)
        config = SARSAnConfig()
        agent = SARSAn(sarsa_config=config)
        with pytest.raises(InvalidParamValue) as e:
            agent.actions_before_training(env=env)

    def test_actions_before_training_throws_estimator_not_set(self):

        env_config = TiledEnvConfig()
        env_config.max_size = 4096
        env_config.num_tilings = 5
        env_config.tiling_dim = 6
        env_config.env = None
        env_config.column_scales = {"col1": [0.0, 1.0]}

        env = TiledEnv(env_config)
        policy = EpsilonGreedyPolicy(eps=1.0, n_actions=1, decay_op=EpsilonDecayOption.NONE)
        config = SARSAnConfig()
        config.policy = policy
        agent = SARSAn(sarsa_config=config)
        with pytest.raises(ValueError) as e:
            agent.actions_before_training(env=env)
            self.assertEqual("Estimator has not been set", str(e))


if __name__ == '__main__':
    unittest.main()
