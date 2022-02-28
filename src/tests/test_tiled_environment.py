"""
Unit-tests for TiledEnv class
"""
import unittest
import pytest
from src.spaces.tiled_environment import TiledEnv, TiledEnvConfig
from src.exceptions.exceptions import InvalidParamValue

class DummyEnv(object):

    def __init__(self):
        self.column_names = ["col1", "col2"]

class TestTiledEnv(unittest.TestCase):

    def test_constructor_raises_zero_max_size(self):
        config = TiledEnvConfig()
        config.env = None
        config.max_size = 0
        config.tiling_dim = 2
        config.num_tilings = 5
        with pytest.raises(InvalidParamValue) as e:
            env = TiledEnv(config)

    def test_constructor_raises_invalid_max_size(self):
        config = TiledEnvConfig()
        config.env = None
        config.max_size = 1
        config.tiling_dim = 2
        config.num_tilings = 5
        with pytest.raises(InvalidParamValue) as e:
            env = TiledEnv(config)

    def test_none_column_ranges(self):
        config = TiledEnvConfig()
        config.env = None
        config.max_size = 4096
        config.tiling_dim = 2
        config.num_tilings = 5
        config.column_ranges = None
        with pytest.raises(InvalidParamValue) as e:
            env = TiledEnv(config)

    def test_empty_column_ranges(self):
        config = TiledEnvConfig()
        config.env = None
        config.max_size = 4096
        config.tiling_dim = 2
        config.num_tilings = 5
        config.column_ranges = {}
        with pytest.raises(InvalidParamValue) as e:
            env = TiledEnv(config)

    def test_create_bins(self):
        config = TiledEnvConfig()
        config.env = DummyEnv()
        config.max_size = 4096
        config.tiling_dim = 2
        config.num_tilings = 2
        config.column_ranges = {"col1": [0.0, 1.0], "col2": [0.0, 1.0]}
        env = TiledEnv(config)
        env.create_bins()

        tiles = env.column_bins
        # we must have as many bins as columns
        self.assertEqual(2, len(tiles))

        for column in tiles:
            # for each column we must have config.num_tilings
            self.assertEqual(config.num_tilings, len(tiles[column]))

            # each tiling must have config.n_bins
            for tile in tiles[column]:
                self.assertEqual(config.n_bins, len(tile))


if __name__ == '__main__':
    unittest.main()
