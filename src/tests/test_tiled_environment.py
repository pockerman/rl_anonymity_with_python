"""Unit-tests for TiledEnv class
"""
import unittest
import pytest
from src.spaces.tiled_environment import TiledEnv, TiledEnvConfig
from src.spaces.discrete_state_environment import DiscreteStateEnvironment
from src.datasets.datasets_loaders import MockSubjectsLoader, MockSubjectsData
from src.spaces.time_step import StepType
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity
from src.spaces.state import State
from src.exceptions.exceptions import InvalidParamValue


class DummyEnv(object):

    def __init__(self):
        self.column_names = ["col1", "col2"]


class TestTiledEnv(unittest.TestCase):

    def load_mock_subjects(self) -> MockSubjectsLoader:
        mock_data = MockSubjectsData(COLUMNS_TYPES={"ethnicity": str, "salary": float, "diagnosis": int},
                                     FEATURES_DROP_NAMES=["NHSno", "given_name", "surname", "dob"] + ["preventative_treatment", "gender",
                                                             "education", "mutation_status"], NORMALIZED_COLUMNS = ["salary"])

        ds = MockSubjectsLoader(mock_data)
        return ds

    def load_discrete_env(self):

        mock_ds = self.load_mock_subjects()
        action_space = ActionSpace.from_actions(ActionIdentity(column_name="ethnicity"),
                                                ActionIdentity(column_name="salary"))
        env = DiscreteStateEnvironment.from_dataset(data_set=mock_ds, action_space=action_space)

        return env

    def test_build_tiles(self):

        discrete_env = self.load_discrete_env()
        tiled_env_config = TiledEnvConfig(n_layers=1, n_bins=10,
                                          env=discrete_env,
                                          column_ranges={"ethnicity": [0.0, 1.0],
                                                         "salary": [0.0, 1.0],
                                                         "diagnosis": [0.0, 1.0]})
        tiled_env = TiledEnv(tiled_env_config)
        tiled_env.create_tiles()

        # should have one layer
        self.assertEqual(1, len(tiled_env.tiles))

        # get the only layer
        layer0 = tiled_env.tiles[0]
        self.assertEqual(discrete_env.n_actions * tiled_env_config.n_bins ** len(tiled_env_config.column_ranges),
                         len(layer0))

    def test_raw_state_index(self):

        discrete_env = self.load_discrete_env()
        tiled_env_config = TiledEnvConfig(n_layers=1, n_bins=10,
                                          env=discrete_env,
                                          column_ranges={"ethnicity": [0.0, 1.0],
                                                         "salary": [0.0, 1.0],
                                                         "diagnosis": [0.0, 1.0]})
        tiled_env = TiledEnv(tiled_env_config)
        tiled_env.create_tiles()

        # should have one layer
        self.assertEqual(1, len(tiled_env.tiles))

        # get the only layer
        layer0 = tiled_env.tiles[0]
        self.assertEqual(discrete_env.n_actions * tiled_env_config.n_bins ** len(tiled_env_config.column_ranges),
                         len(layer0))

        raw_state = State()
        raw_state.column_distortions = {"ethnicity": 0.01, "salary": 0.01, "diagnosis": 0.01}

        global_index = layer0.get_global_tile_index(raw_state=raw_state, action=0)

        # that's should be in the lower left
        # cube
        self.assertEqual(0, global_index)

        # create another raw state
        raw_state.column_distortions = {"ethnicity": 0.01, "salary": 0.15, "diagnosis": 0.01}

        global_index = layer0.get_global_tile_index(raw_state=raw_state, action=0)
        self.assertEqual(10, global_index)

    def test_none_column_ranges(self):
        config = TiledEnvConfig()
        config.env = None
        config.n_bins = 2
        config.n_layers = 5
        config.column_ranges = None
        with pytest.raises(InvalidParamValue) as e:
            env = TiledEnv(config)

    def test_empty_column_ranges(self):
        config = TiledEnvConfig()
        config.env = None
        config.n_bins = 2
        config.n_layers = 5
        config.column_ranges = {}
        with pytest.raises(InvalidParamValue) as e:
            env = TiledEnv(config)

    def test_reset(self):

        discrete_env = self.load_discrete_env()

        config = TiledEnvConfig()
        config.env = discrete_env
        config.n_bins = 2
        config.n_layers = 2
        config.column_ranges = {}
        for col in discrete_env.column_names:
            config.column_ranges[col] = [0.0, 1.0]

        env = TiledEnv(config)
        env.create_tiles()
        time_step = env.reset()

        self.assertEqual(StepType.FIRST, time_step.step_type)

        tiled_observation = time_step.observation

        self.assertEqual(config.n_layers * (config.n_bins ** len(config.column_ranges)) * env.n_actions,
                         tiled_observation.shape[0])

        # we should not have any distortion
        # so the (1, 1, 1) local index should be in the
        # first item
        self.assertEqual(1.0, tiled_observation[0])


if __name__ == '__main__':
    unittest.main()


