"""
Unit-tests for class Trainer
"""
import unittest
import pytest

from src.trainers.trainer import Trainer
from src.spaces.tiled_environment import TiledEnv


class TestTrainer(unittest.TestCase):

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_with_sarsa_semi_grad_agent(self):
        pass

if __name__ == '__main__':
    unittest.main()
