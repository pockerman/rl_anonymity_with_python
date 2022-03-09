"""
Tile environment
"""

import copy
from typing import TypeVar, List, Any
from dataclasses import dataclass

import numpy as np

from src.extern.tile_coding import IHT, tiles
from src.spaces.actions import ActionBase, ActionType
from src.spaces.time_step import TimeStep
from src.exceptions.exceptions import InvalidParamValue
from src.spaces.state import State
from src.spaces.time_step import copy_time_step

Env = TypeVar('Env')
Config = TypeVar('Config')
RawState = TypeVar('RawState')
TiledState = TypeVar('TiledState')
Action = TypeVar('Action')

INVALID_ID = -1


@dataclass(init=True, repr=True)
class TiledEnvConfig(object):
    """Configuration for the TiledEnvironment
    """

    env: Env = None
    n_layers: int = 1
    n_bins: int = 1
    column_ranges: dict = None


class Tile(object):
    """Helper class that models a tile

    """
    def __init__(self, global_id: int):
        self.global_idx: int = global_id
        self.columns_bins = {}

    def build(self, bin_indices: dir, n_bins: int):

        touched = False
        # how many bins
        for column in bin_indices:

            if bin_indices[column] < n_bins:
                self.columns_bins[column] = bin_indices[column]

                # increase the index
                bin_indices[column] += 1

                # the rest of the columns remain the same
                for column2 in bin_indices:

                    if column2 != column:
                        self.columns_bins[column2] = bin_indices[column2]

                break
            else:

                if bin_indices[column] != n_bins:
                    raise ValueError("Invalid number of bins={0} "
                                     "should be equal to {1}".format(bin_indices[column], n_bins))

                self.columns_bins[column] = bin_indices[column]

    def search(self, bin_indices):
        """
        Returns true if the given bin indices are represented
        by the tile

        Parameters
        ----------
        bin_indices: The list of bin indices

        Returns
        -------
        boolean


        """

        if len(self.columns_bins) == 0:
            raise InvalidParamValue(param_name="column_bins", param_value="0. Column bins is empty")

        if len(bin_indices) != len(self.columns_bins):
            raise ValueError("len(bin_indices) = {0} != len( self.columns_bins) = {1}".format(len(bin_indices),
                                                                                              len(self.columns_bins)))

        col_indices = []
        for col in self.columns_bins:
            col_indices.append((col, self.columns_bins[col]))

        if col_indices == bin_indices:
            return True

        return False


class Layer(object):
    """Helper class to represent a layer of tiling

    """

    @staticmethod
    def n_tiles_per_action(n_bins: int, n_columns: int) -> int:
        return n_bins ** n_columns

    def __init__(self, column_bins, n_bins: int,
                 n_actions: int, start_index: int, end_index: int):
        self.column_bins = column_bins
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.start_index = start_index
        self.end_index = end_index
        self.tiles = {}

    def __len__(self):
        return len(self.tiles)

    def build_tiles(self, next_tile_global_idx: int) -> int:
        """Build the tiles for the layer. For each action creates
        self.n_bins ** len(self.column_bins) tiles.

        Parameters
        ----------

        next_tile_global_idx: The starting point of global tile ids for the layer

        Returns
        -------

        The next tile global id

        """

        # number of total tiles we have in th
        # the layer
        n_total_tiles = self.n_bins ** len(self.column_bins)
        for action in range(self.n_actions):
            bin_indices = {key: 0 for key in self.column_bins}
            next_local_tile_idx = 0
            next_local_tile_idx, next_tile_global_idx = self._do_build_tile(action=action, next_local_tile_idx=next_local_tile_idx,
                                next_tile_global_idx=next_tile_global_idx)

            """
            key_list = list(bin_indices.keys())
            for t in range(n_total_tiles):
                self.tiles[(action, t)] = Tile(global_id=next_tile_global_idx)
                self.tiles[(action, t)].build(bin_indices, self.n_bins)

                # check if we have reached the number of bins
                # if yes zero everything and increment the
                # next index

                #next_tile_global_idx += 1
            """
        return next_tile_global_idx


    def _do_build_tile(self, action: int, next_local_tile_idx: int,
                                next_tile_global_idx: int) -> tuple:

        if len(self.column_bins) != 3:
            raise NotImplementedError("This function is not implemented for more than three columns")

        if len(self.column_bins) == 3:
            return self._do_build_three_columns(action=action, next_local_tile_idx=next_local_tile_idx,
                                                next_tile_global_idx=next_tile_global_idx)

    def _do_build_three_columns(self, action: int, next_local_tile_idx: int,
                                next_tile_global_idx: int) -> tuple:

        key_list = list(self.column_bins.keys())

        # bin indices in np.digitize start at one
        for bi in range(1, self.n_bins + 1):
            for bj in range(1, self.n_bins + 1):
                for bk in range(1, self.n_bins + 1):
                    self.tiles[(action, next_local_tile_idx)] = Tile(global_id=next_tile_global_idx)
                    self.tiles[(action, next_local_tile_idx)].columns_bins = {key_list[0]: bi, key_list[1]: bj, key_list[2]: bk}
                    next_local_tile_idx += 1
                    next_tile_global_idx += 1

        return next_local_tile_idx, next_tile_global_idx

    def get_global_tile_index(self, raw_state: RawState, action: Action) -> int:
        """Returns the global tile index for the raw state and the given action
        If the bin indices corresponding to the raw state after digitization
        cannot be found in any tile then it returns -1

        Parameters
        ----------
        raw_state: The raw state to digitize
        action: The action taken

        Returns
        -------

        The global tile index
        """

        # get the bin indices in the layer for the
        # raw_state
        bin_indices = [(name, np.digitize(raw_state.column_distortions[name],
                                          self.column_bins[name])) for name in raw_state.column_distortions]

        global_tile_idx = INVALID_ID

        for _, t in self.tiles:
            tile = self.tiles[(action, t)]

            # check if the bin indices
            # that correspond to the raw state tiling
            # are in the given tile. If yes return the
            # tile global index
            if tile.search(bin_indices):
                global_tile_idx = tile.global_idx
                break

        return global_tile_idx


class Tiles(object):
    """Helper class for tile manipulation. It holds a list
    of layers. For every layer creates

    """

    def __init__(self, n_layers: int, n_bins: int, n_actions: int, column_ranges: dict):
        """Constructor. Initialize by passing the number of layers, number of bins,
        number of actions and the column ranges

        Parameters

        ----------

        n_layers: number of layers to use
        n_bins: Number of bins to use per column
        n_actions: Number of actions allowed in the environment
        column_ranges: The range of values that each column can take

        """
        self.layers = {}
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.column_ranges = column_ranges

    def __getitem__(self, layer) -> Layer:
        return self.layers[layer]

    def __len__(self):
        return len(self.layers)

    def build(self) -> None:
        """Build the tiles layers

        Returns
        -------
        None

        """

        start_layer_idx = 0
        next_tile_global_idx = 0
        for layer in range(self.n_layers):

            # for each layer we compute the column bins
            column_bins = {}
            for column in self.column_ranges:
                range_ = self.column_ranges[column]
                tile_width = (range_[1] + range_[0]) / self.n_bins
                column_bins[column] = np.linspace(range_[0] + layer * tile_width,
                                                  range_[1] + layer * tile_width,
                                                  self.n_bins)

            end_layer_idx = (layer + 1)*(self.n_bins ** len(self.column_ranges)) * self.n_actions
            # now create the layer
            new_layer = Layer(column_bins=column_bins, n_bins=self.n_bins,
                              n_actions=self.n_actions,
                              start_index=start_layer_idx,
                              end_index=end_layer_idx)

            # build the tiles for the new layer
            next_tile_global_idx = new_layer.build_tiles(next_tile_global_idx)
            self.layers[layer] = new_layer

            start_layer_idx = end_layer_idx


class TiledEnv(object):
    """The TiledEnv class. It models a tiled
    environment
    """

    IS_TILED_ENV_CONSTRAINT = True

    @classmethod
    def from_options(cls, *, env: Env,  n_layers: int,
                    n_bins: int, column_ranges: dict):
        return cls(TiledEnvConfig(env=env,
                                  n_layers=n_layers,
                                  n_bins=n_bins, column_ranges=column_ranges))

    def __init__(self, config: TiledEnvConfig) -> None:

        self.env = config.env
        self.n_layers = config.n_layers
        self.n_bins = config.n_bins

        # set up the columns scaling
        # only the columns that are to be altered participate in the
        # tiling
        self.column_ranges = config.column_ranges
        self.column_scales = {}
        self.tiles: Tiles = None

        # Initialize index hash table (IHT) for tile coding.
        # This assigns a unique index to each tile up to max_size tiles.
        self._validate()
        self._create_column_scales()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def n_actions(self) -> int:
        return len(self.env.action_space)

    @property
    def n_states(self) -> int:
        """Returns the total number of states in the environment

        Returns
        -------

        The total number of states in the environment

        """
        return self.n_layers * Layer.n_tiles_per_action(self.n_bins, len(self.column_ranges))

    @property
    def config(self) -> Config:
        return self.env.config

    def step(self, action: ActionBase, **options) -> TimeStep:
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

        if "tiled_state" in options and options['tiled_state'] is True:

            # we want to put the state into the tiles
            tiled_state = self.featurize_raw_state(state)
            time_step = copy_time_step(time_step=time_step, **{"observation": tiled_state})

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

        if self.tiles is None or len(self.tiles) == 0:
            raise InvalidParamValue(param_name="tiles", param_value="{}. Have you called create_tiles?")

        # reset the raw environment
        raw_time_step = self.env.reset(**options)

        # a state wrapper to communicate
        state = State()

        # the raw environment returns an index
        # of the bin that the total distortion falls into
        state.bin_idx = raw_time_step.observation
        state.total_distortion = raw_time_step.info["total_distortion"]
        state.column_distortions = self.env.column_distortions

        time_step = copy_time_step(time_step=raw_time_step, **{"observation": state})

        if "tiled_state" in options and options['tiled_state'] is True:

            # we want to put the state into the tiles
            tiled_state = self.featurize_raw_state(state)
            time_step = copy_time_step(time_step=time_step, **{"observation": tiled_state})

        return time_step

    def get_state_action_tile_matrix(self, state: TiledState) -> np.array:
        """ Transform the TiledState vector to a numpy 2D array

        Parameters
        ----------

        state: The tiled state-action vector

        Returns
        -------

        A 2d numpy array
        """

        return state.reshape(self.n_layers, self.n_actions, Layer.n_tiles_per_action(n_bins=self.n_bins,
                                                                                     n_columns=len(self.column_ranges)))

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

    def create_tiles(self) -> None:
        """Create the bins

        Returns
        -------

        None

        """

        # calculate the tile width for each column in the
        # data set
        self.tiles = Tiles(n_bins=self.n_bins, n_layers=self.n_layers,
                           n_actions=self.n_actions, column_ranges=self.column_ranges)
        self.tiles.build()

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

    def featurize_state_action(self, state: RawState, action: ActionBase) -> TiledState:
        """Get a list of Tiles for the given state and action

        Parameters
        ----------
        state: The environment state observed
        action: The action

        Returns
        -------

        A list of tiles

        """

        tiled_state = np.zeros(self.n_layers * self.n_actions * self.n_bins ** (len(self.column_ranges)))

        found = False
        for layer in range(self.n_layers):
            global_idx = self.tiles[layer].get_global_tile_index(raw_state=state, action=action)
            if global_idx != INVALID_ID:
                tiled_state[global_idx] = 1.0

        return tiled_state

    def featurize_raw_state(self, state: RawState) -> TiledState:
        """Returns the tiled state vector given  the vector
        of column distortions

        Parameters
        ----------
        state

        Returns
        -------

        """

        tiled_state = np.zeros(self.n_layers * self.n_actions * self.n_bins ** (len(self.column_ranges)))

        found = False
        for layer in range(self.n_layers):

            #if found:
             #   break

            for action in range(self.n_actions):
                global_idx = self.tiles[layer].get_global_tile_index(raw_state=state, action=action)
                if global_idx != INVALID_ID:
                    tiled_state[global_idx] = 1.0
                    #found = True
                    #break

        return tiled_state

    def _create_column_scales(self) -> None:
        """
        Create the scales for each column

        Returns
        -------

        None

        """

        for name in self.column_ranges:
            range_ = self.column_ranges[name]
            self.column_scales[name] = self.n_bins / (range_[1] - range_[0])

    def _validate(self) -> None:
        """
        Validate the internal data structures

        Returns
        -------

        None

        """

        """
        if self.max_size <= 0:
            raise InvalidParamValue(param_name="max_size",
                                    param_value=str(self.max_size) + " should be > 0")

        # Ensure max_size >= total number of tiles (num_tilings x tiling_dim x tiling_dim)
        # to ensure no duplicates.
        if self.max_size < self.n_layers * self.tiling_dim * self.tiling_dim:
            raise InvalidParamValue(param_name="max_size",
                                    param_value=str(self.max_size) +
                                                " should be >=num_tilings * tiling_dim * tiling_dim")
        """

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

        if self.n_layers == 0:
            raise InvalidParamValue(param_name="n_layers",
                                    param_value=str(len(self.column_scales)) + " n_layers cannot be zero")

