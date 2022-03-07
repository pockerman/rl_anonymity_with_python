import unittest
import pytest

from src.algorithms.semi_gradient_sarsa import SemiGradSARSAConfig, SemiGradSARSA
from src.algorithms.epsilon_greedy_q_estimator import EpsilonGreedyQEstimator
from src.exceptions.exceptions import InvalidParamValue
from src.spaces.tiled_environment import TiledEnv
from src.spaces.discrete_state_environment import DiscreteStateEnvironment
from src.datasets.datasets_loaders import MockSubjectsLoader, MockSubjectsData

class TestSemiGradSARSA(unittest.TestCase):

    def test_constructor(self):
        config = SemiGradSARSAConfig()
        semi_grad_sarsa = SemiGradSARSA(config)
        self.assertIsNotNone(semi_grad_sarsa.config)

    def test_actions_before_training_throws_1(self):

        semi_grad_sarsa = SemiGradSARSA(None)
        with pytest.raises(InvalidParamValue) as e:
            semi_grad_sarsa.actions_before_training(env=None)

    def test_actions_before_training_throws_2(self):
        config = SemiGradSARSAConfig()
        config.n_itrs_per_episode = 0
        semi_grad_sarsa = SemiGradSARSA(config)

        # make sure this is valid
        self.assertIsNotNone(semi_grad_sarsa.config)

        with pytest.raises(ValueError) as e:
            semi_grad_sarsa.actions_before_training(env=None)

    def test_actions_before_training_throws_3(self):
        config = SemiGradSARSAConfig()
        semi_grad_sarsa = SemiGradSARSA(config)

        # make sure this is valid
        self.assertIsNotNone(semi_grad_sarsa.config)

        with pytest.raises(InvalidParamValue) as e:
            semi_grad_sarsa.actions_before_training(env=None)

    @pytest.mark.skip(reason="env cannot be None")
    def test_on_episode_returns_info(self):
        config = SemiGradSARSAConfig()
        semi_grad_sarsa = SemiGradSARSA(config)

        # make sure this is valid
        self.assertIsNotNone(semi_grad_sarsa.config)

        episode_info = semi_grad_sarsa.on_episode(env=None)
        self.assertIsNotNone(episode_info)

    def test_on_episode_trains(self):

        sarsa_config = SemiGradSARSAConfig(n_itrs_per_episode=1, policy=EpsilonGreedyQEstimator())
        semi_grad_sarsa = SemiGradSARSA(sarsa_config)

        # cretate a default data
        ds_default_data = MockSubjectsData()
        ds = MockSubjectsLoader.from_options(filename=ds_default_data.FILENAME,
                                             names=ds_default_data.NAMES, drop_na=ds_default_data.DROP_NA,
                                             change_col_vals=ds_default_data.CHANGE_COLS_VALS,
                                             features_drop_names=ds_default_data.FEATURES_DROP_NAMES +
                                                                 ["preventative_treatment", "gender",
                                                                  "education", "mutation_status"],
                                             column_normalization=["salary"], column_types={"ethnicity": str,
                                                                                            "salary": float,
                                                                                            "diagnosis": int})

        discrete_env = DiscreteStateEnvironment.from_options(data_set=ds, action_space=None,
                                                             reward_manager=None, distortion_calculator=None)
        tiled_env = TiledEnv.from_options(env=discrete_env, max_size=4096, num_tilings=5, n_bins=10,
                                          column_ranges={"ethnicity": [0.0, 1.0],
                                                         "salary": [0.0, 1.0],
                                                         "diagnosis": [0.0, 1.0]}, tiling_dim=3)

        """
        # specify the columns to drop
        drop_columns = MockSubjectsLoader.FEATURES_DROP_NAMES + ["preventative_treatment", "gender",
                                                                 "education", "mutation_status"]
        MockSubjectsLoader.FEATURES_DROP_NAMES = drop_columns

        # do a salary normalization so that we work with
        # salaries in [0, 1] this is needed as we will
        # be using normalized distances
        MockSubjectsLoader.NORMALIZED_COLUMNS = ["salary"]

        # specify the columns to use
        MockSubjectsLoader.COLUMNS_TYPES = {"ethnicity": str, "salary": float, "diagnosis": int}
        ds = MockSubjectsLoader()
        """

        # create the discrete environment

        semi_grad_sarsa.actions_before_training(tiled_env)

        # make sure this is valid
        self.assertIsNotNone(semi_grad_sarsa.config)

        episode_info = semi_grad_sarsa.on_episode(env=tiled_env)
        self.assertIsNotNone(episode_info)


if __name__ == '__main__':
    unittest.main()