"""
Unit-tests for TiledEnv class
"""
import unittest
import pytest
from src.spaces.tiled_environment import TiledEnv, TiledEnvConfig
from src.exceptions.exceptions import InvalidParamValue


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

    def test_empty_column_scales(self):
        config = TiledEnvConfig()
        config.env = None
        config.max_size = 4096
        config.tiling_dim = 2
        config.num_tilings = 5
        config.columns_scales = {}
        with pytest.raises(InvalidParamValue) as e:
            env = TiledEnv(config)



if __name__ == '__main__':
    unittest.main()
