"""
Tile environment
"""

from typing import TypeVar
from src.extern.tile_coding import IHT, tiles
from src.spaces.actions import ActionBase, ActionType
from src.spaces.time_step import TimeStep

Env = TypeVar('Env')
State = TypeVar('State')


class TiledEnv(object):

    IS_TILED_ENV_CONSTRAINT = True

    def __init__(self, env: Env, num_tilings: int, max_size: int,
                 tiling_dim: int) -> None:

        self.env = env
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = tiling_dim

        # Initialize index hash table (IHT) for tile coding.
        # This assigns a unique index to each tile up to max_size tiles.
        # Ensure max_size >= total number of tiles (num_tilings x tiling_dim x tiling_dim)
        # to ensure no duplicates.
        self.iht = IHT(max_size)

        # set up the columns scaling
        # only the columns that are to be altered participate in the
        # tiling
        self.columns_scales = {}

    def step(self, action: ActionBase) -> TimeStep:
        """
         Apply the action and return new state
        :param action: The action to apply
        :return:
        """

        pass

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def n_actions(self) -> int:
        return len(self.env.action_space)

    @property
    def n_states(self) -> int:
        return self.env.n_states

    def get_action(self, aidx: int) -> ActionBase:
        return self.env.action_space[aidx]

    def save_current_dataset(self, episode_index: int, save_index: bool = False) -> None:
        """
        Save the current distorted datase for the given episode index
        :param episode_index:
        :param save_index:
        :return:
        """
        self.env.save_current_dataset(episode_index, save_index)

    def create_bins(self) -> None:
        """
        Create the bins
        :return:
        """
        self.env.create_bins()

    def get_aggregated_state(self, state_val: float) -> int:
        """
        Returns the bin index that the state_val corresponds to
        :param state_val: The value of the state. This typically will be
        either a column normalized distortion value or the dataset average total
        distortion
        :return:
        """
        return self.env.get_aggregated_state(state_val)

    def initialize_column_counts(self) -> None:
        """
        Set the column visit counts to zero
        :return:
        """
        self.env.initialize_column_counts()

    def all_columns_visited(self) -> bool:
        """
        Returns True is all column counts are greater than zero
        :return:
        """
        return self.env.all_columns_visited()

    def initialize_distances(self) -> None:
        """
        Initialize the text distances for features of type string. We set the
        normalized distance to 0.0 meaning that no distortion is assumed initially
        :return: None
        """
        self.env.initialize_distances()

    def apply_action(self, action: ActionBase):
        """
        Apply the action on the environment
        :param action: The action to apply on the environment
        :return:
        """
        self.env.apply_action(action)

    def total_current_distortion(self) -> float:
        """
        Calculates the current total distortion of the dataset.
        :return:
        """

        return self.env.total_current_distortion()

    def reset(self, **options) -> TimeStep:
        """
        Starts a new sequence and returns the first `TimeStep` of this sequence.
        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
              Scalar values that can be cast to NumPy arrays (e.g. Python floats)
              are also valid in place of a scalar array. Must conform to the
              specification returned by `observation_spec()`.
        """
        return self.env.reset(**options)

    def get_scaled_state(self, state: State) -> list:
        """
        Scales the state components ad returns the
        scaled state
        :param state:
        :return:
        """
        scaled_state_vals = []
        for name in state:

            scaled_state_vals.append(state[name] * self.columns_scales[name])

        return scaled_state_vals

    def featurize_state_action(self, state, action: ActionBase) -> None:
        """
        Returns the featurized representation for a state-action pair
        :param state:
        :param action:
        :return:
        """
        scaled_state = self.get_scaled_state(state)
        featurized = tiles(self.iht, self.num_tilings, scaled_state, [action])
        return featurized







