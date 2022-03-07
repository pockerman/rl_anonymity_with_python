"""
Unit-tests for class Trainer
"""
import unittest
import pytest

from src.algorithms.trainer import Trainer
from src.algorithms.n_step_semi_gradient_sarsa import SARSAnConfig, SARSAn
from src.spaces.tiled_environment import TiledEnv


class TestTrainer(unittest.TestCase):

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_with_sarsa_semi_grad_agent(self):

        # create tiled environment
        tiled_env = TiledEnv(env=None, num_tilings=10, max_size=4096,
                             tiling_dim=5)

        sarsa_config = SARSAnConfig()
        agent = SARSAn(sarsa_config=sarsa_config)

        trainer = Trainer(agent=agent, env=tiled_env,
                          configuration={"n_episodes": 1})

        trainer.train()


if __name__ == '__main__':
    unittest.main()
