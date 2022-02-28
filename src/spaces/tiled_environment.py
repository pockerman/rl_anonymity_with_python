"""
Tile environment
"""

import copy
from typing import TypeVar, List
from dataclasses import dataclass

import numpy as np

from src.extern.tile_coding import IHT, tiles
from src.spaces.actions import ActionBase, ActionType
from src.spaces.time_step import TimeStep
from src.exceptions.exceptions import InvalidParamValue
from src.spaces.state import State
from src.spaces.time_step import copy_time_step

Env = TypeVar('Env')
Tile = TypeVar('Tile')
Config = TypeVar('Config')


@dataclass(init=True, repr=True)
class TiledEnvConfig(object):
    """Configuration for the TiledEnvironment
    """

    env: Env = None
    num_tilings: int = 0
    max_size: int = 0
    tiling_dim: int = 0
    n_bins: int = 1
    column_ranges: dict = None


class TiledEnv(object):
    """The TiledEnv class. It models a tiled
    environment
    """

    IS_TILED_ENV_CONSTRAINT = True

    @classmethod
    def from_options(cls, *, env: Env, max_size: int, num_tilings: int,
                     tiling_dim: int, n_bins: int, column_ranges: dict):
        return cls(TiledEnvConfig(env=env, max_size=max_size, num_tilings=num_tilings,
                                  tiling_dim=tiling_dim, n_bins=n_bins, column_ranges=column_ranges))

    def __init__(self, config: TiledEnvConfig) -> None:

        self.env = config.env
        self.max_size = config.max_size
        self.num_tilings = config.num_tilings
        self.tiling_dim = config.tiling_dim
        self.n_bins = config.n_bins

        # set up the columns scaling
        # only the columns that are to be altered participate in the
        # tiling
        self.column_ranges = config.column_ranges
        self.column_scales = {}

        # Initialize index hash table (IHT) for tile coding.
        # This assigns a unique index to each tile up to max_size tiles.
        self._validate()
        self._create_column_scales()
        self.iht = IHT(self.max_size)

        self.column_bins = {}

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def n_actions(self) -> int:
        return len(self.env.action_space)

    @property
    def n_states(self) -> int:
        return self.env.n_states

    @property
    def config(self) -> Config:
        return self.env.config

    def step(self, action: ActionBase) -> TimeStep:
        """Execute the action in the environment and return
        a new state for observation

        Parameters
        ----------
        action: The action to execute

        Returns
        -------

         An instance of TimeStep type

        """

        raw_time_step = self.env.step(action)

        # a state wrapper to communicate
        state = State()

        # the raw environment returns an index
        # of the bin that the total distortion falls into
        state.bin_idx = raw_time_step.observation
        state.total_distortion = raw_time_step.info["total_distortion"]
        state.column_distortions = self.env.column_distortions

        time_step = copy_time_step(time_step=raw_time_step, **{"observation": state})
        return time_step

    def reset(self, **options) -> TimeStep:
        """Reset the environment so that a new sequence
        of episodes can be generated

        Parameters
        ----------
        options: Client provided named options

        Returns
        -------

        An instance of TimeStep type
        """

        raw_time_step = self.env.reset(**options)

        # a state wrapper to communicate
        state = State()

        # the raw environment returns an index
        # of the bin that the total distortion falls into
        state.bin_idx = raw_time_step.observation
        state.total_distortion = raw_time_step.info["total_distortion"]
        state.column_distortions = self.env.column_distortions

        time_step = copy_time_step(time_step=raw_time_step, **{"observation": state})

        return time_step

    def get_action(self, aidx: int) -> ActionBase:
        """Returns the action that corresponds to the given index

        Parameters
        ----------
        aidx: The index of the action to return

        Returns
        -------

        An instance of the ActionBase class

        """
        return self.env.action_space[aidx]

    def save_current_dataset(self, episode_index: int, save_index: bool = False) -> None:
        """Save the current data set at the given episode index

        Parameters
        ----------

        episode_index: Episode index corresponding to the training episode
        save_index: if True the Pandas index is also saved

        Returns
        -------

        None

        """
        self.env.save_current_dataset(episode_index, save_index)

    def create_bins(self) -> None:
        """Create the bins

        Returns
        -------

        None

        """

        # calculate the tile width for each column in the
        # data set

        tile_widhs = {}
        for column in self.column_ranges:
            range_ = self.column_ranges[column]
            tile_width = (range_[1] + range_[0]) / self.n_bins
            self.column_bins[column] = np.zeros((self.num_tilings, self.n_bins))

            # for each layer create an offset
            # bin
            for i in range(self.num_tilings):
                self.column_bins[column][i] = np.linspace(range_[0] + i * tile_width, range_[1] + i * tile_width, self.n_bins)

    def get_aggregated_state(self, state_val: float) -> int:
        """
        Returns the bin index that the state_val corresponds to

        Parameters
        ----------

        state_val: The bin index that the distortion corresponds to

        Returns
        -------

        The bin index corresponding to the distortion
        """
        return self.env.get_aggregated_state(state_val)

    def initialize_column_counts(self) -> None:
        """
        Set the column visit counts to zero
        Returns
        -------

        None

        """
        self.env.initialize_column_counts()

    def all_columns_visited(self) -> bool:
        """
        Returns True is all column counts are greater than zero

        Returns
        -------

        Returns True is all column counts are greater than zero
        otherwise False

        """
        return self.env.all_columns_visited()

    def initialize_distances(self) -> None:
        """
        Initialize the text distances for features of type string. We set the
        normalized distance to 0.0 meaning that no distortion is assumed initially

        Returns
        -------

        None

        """
        self.env.initialize_distances()

    def apply_action(self, action: ActionBase) -> None:
        """Apply the action on the environment

        Parameters
        ----------
        action: The action to apply

        Returns
        -------

        None

        """
        self.env.apply_action(action)

    def total_current_distortion(self) -> float:
        """Calculates the current total distortion of the dataset.

        Returns
        -------
        The total current distortion

        """
        return self.env.total_current_distortion()

    def get_scaled_state(self, state: State) -> list:
        """Scales the state components and returns the
        scaled state

        Parameters
        ----------
        state: The state to scale

        Returns
        -------

        A list of scaled state values

        """
        scaled_state_vals = []
        for name in state:
            scaled_state_vals.append(state[name] * self.column_scales[name])

        return scaled_state_vals

    def featurize_state_action(self, state: State, action: ActionBase) -> List[Tile]:
        """Get a list of Tiles for the given state and action

        Parameters
        ----------
        state: The environment state observed
        action: The action

        Returns
        -------

        A list of tiles

        """

        scaled_state = self.get_scaled_state(state)
        featurized = tiles(self.iht, self.num_tilings, scaled_state, [action])
        return featurized

    def _create_column_scales(self) -> None:
        """
        Create the scales for each column

        Returns
        -------

        None

        """

        for name in self.column_ranges:
            range_ = self.column_ranges[name]
            self.column_scales[name] = self.tiling_dim / (range_[1] - range_[0])

    def _validate(self) -> None:
        """
        Validate the internal data structures

        Returns
        -------

        None

        """
        if self.max_size <= 0:
            raise InvalidParamValue(param_name="max_size",
                                    param_value=str(self.max_size) + " should be > 0")

        # Ensure max_size >= total number of tiles (num_tilings x tiling_dim x tiling_dim)
        # to ensure no duplicates.
        if self.max_size < self.num_tilings * self.tiling_dim * self.tiling_dim:
            raise InvalidParamValue(param_name="max_size",
                                    param_value=str(self.max_size) +
                                    " should be >=num_tilings * tiling_dim * tiling_dim")

        if self.column_ranges is None:
            raise InvalidParamValue(param_name="column_ranges",
                                    param_value="None")

        if len(self.column_ranges) == 0:
            raise InvalidParamValue(param_name="column_scales",
                                    param_value=str(len(self.column_scales)) + " should not be empty")

        if self.env is None:
            raise InvalidParamValue(param_name="env",
                                    param_value="None")

        if len(self.column_ranges) != len(self.env.column_names):
            raise ValueError("Column ranges is not equal to number of columns")

