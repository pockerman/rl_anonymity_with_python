"""
RL Environment API taken from
https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""

import copy
import numpy as np
import torch
from pathlib import Path
from typing import TypeVar, List, Any
from dataclasses import dataclass

from src.spaces.env_type import DiscreteEnvType
from src.spaces.actions import ActionBase, ActionType
from src.spaces.time_step import TimeStep, StepType
from src.datasets import ColumnType
from src.spaces.actions import ActionTransform

DataSet = TypeVar("DataSet")
RewardManager = TypeVar("RewardManager")
ActionSpace = TypeVar("ActionSpace")
DistortionCalculator = TypeVar('DistortionCalculator')


@dataclass(init=True, repr=True)
class DiscreteEnvConfig(object):
    """Configuration for discrete environment
    """

    data_set: DataSet = None
    action_space: ActionSpace = None
    reward_manager: RewardManager = None
    average_distortion_constraint: float = 0.0
    gamma: float = 0.99
    n_states: int = 10
    min_total_distortion: float = 0.4
    min_distortion: Any = 0.4
    max_distortion: Any = 0.7
    max_total_distortion: float = 0.7
    n_rounds_below_min_distortion: int = 10
    distorted_set_path: Path = None
    distortion_calculator: DistortionCalculator = None
    env_type: DiscreteEnvType = DiscreteEnvType.TOTAL_DISTORTION_STATE
    column_types: dict = None
    use_identifying_column_dist_in_total_dist: bool = True
    use_identifying_column_dist_factor: float = 1.0
    state_as_distances: bool = False


class DiscreteStateEnvironment(object):
    """The DiscreteStateEnvironment class. Uses state aggregation in order
    to create bins where the average total distortion of the dataset falls in
    """

    IS_TILED_ENV_CONSTRAINT = False

    @classmethod
    def from_options(cls, *, data_set: DataSet, action_space: ActionSpace,
                     reward_manager: RewardManager, distortion_calculator: DistortionCalculator,
                     average_distortion_constraint: float = 0.0,
                     gamma: float = 0.99, n_states: int = 10,
                     min_distortion: Any = 0.4, min_total_distortion: float = 0.4,
                     max_distortion: Any = 0.7, max_total_distortion: float = 0.7,
                     n_rounds_below_min_distortion: int = 10,
                     env_type: DiscreteEnvType = DiscreteEnvType.TOTAL_DISTORTION_STATE,
                     distorted_set_path: Path = None, column_types: dir={},
                     use_identifying_column_dist_in_total_dist: bool = True,
                     use_identifying_column_dist_factor: float = 1.0,
                     state_as_distances: bool = False):

        config = DiscreteEnvConfig(data_set=data_set, action_space=action_space,
                                   reward_manager=reward_manager,
                                   distortion_calculator=distortion_calculator,
                                   distorted_set_path=distorted_set_path,
                                   n_rounds_below_min_distortion=n_rounds_below_min_distortion,
                                   max_distortion=max_distortion,
                                   max_total_distortion=max_total_distortion,
                                   gamma=gamma,
                                   n_states=n_states, min_distortion=min_distortion,
                                   min_total_distortion=min_total_distortion,
                                   average_distortion_constraint=average_distortion_constraint,
                                   env_type=env_type, column_types=column_types,
                                   use_identifying_column_dist_in_total_dist=use_identifying_column_dist_in_total_dist,
                                   use_identifying_column_dist_factor=use_identifying_column_dist_factor,
                                   state_as_distances=state_as_distances)

        return cls(env_config=config)

    @classmethod
    def from_dataset(cls, data_set: DataSet, *,  action_space: ActionSpace = None,
                     reward_manager: RewardManager = None, distortion_calculator: DistortionCalculator = None):

        config = DiscreteEnvConfig(data_set=data_set, action_space=action_space, reward_manager=reward_manager,
                                   distortion_calculator=distortion_calculator)
        return cls(env_config=config)

    def __init__(self, env_config: DiscreteEnvConfig) -> None:
        self.config = env_config
        self.n_rounds_below_min_distortion = 0
        self.state_bins: List[float] = []
        self.state_space: List[tuple] = []
        self.distorted_data_set = copy.deepcopy(self.config.data_set)
        self.current_time_step: TimeStep = None

        # dictionary that holds the distortion for every column
        # in the dataset
        self.column_distances = {}

        # holds the discretization of [0.0, 1.0]
        # for every column in the dataset. Only filled
        # if config.state_type = MULTI_COLUMN_STATE
        self.column_bins = {}

        # hold a copy of the visits per
        # column. An episode ends when all columns
        # have been visited
        self.column_visits = {}
        self.create_bins()

    @property
    def columns_attribute_types(self) -> dict:
        return self.config.data_set.columns_attribute_types

    @property
    def action_space(self):
        return self.config.action_space

    @property
    def n_actions(self) -> int:
        return len(self.config.action_space)

    @property
    def n_states(self) -> int:
        return self.config.n_states

    @property
    def column_names(self) -> list:
        return self.config.data_set.get_columns_names()

    @property
    def column_distortions(self) -> dict:
        return self.column_distances

    @property
    def env_type(self) -> DiscreteEnvType:
        return self.config.env_type

    def close(self, **kwargs) -> None:
        pass

    def n_quasi_identifying_columns(self) -> int:
        """Returns the number of quasi identifying columns
        assumed in the data set

        Returns
        -------

        The number of quasi identifying columns
        assumed in the data set
        """

        counter = 0
        for name in self.column_names:

            # we create bins only for the QUASI_IDENTIFYING_ATTRIBUTE
            # attributes
            if self.config.column_types[name] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE:
                counter += 1
        return counter

    def is_quasi_identifying_column(self, col_name: str) -> bool:
        """Returns true if the column is a quasi identifying column

        Parameters
        ----------
        col_name: The column name to query

        Returns
        -------

        Returns true if the column is a quasi identifying column
        """
        return self.config.column_types[col_name] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE

    def get_action(self, aidx: int) -> ActionBase:
        """Returns the action if the global aidx index

        Parameters
        ----------

        aidx: The index of the action to return

        Returns
        -------

        An instance of ActionBase

        """
        return self.config.action_space[aidx]

    def save_current_dataset(self, episode_index: int, save_index: bool = False) -> None:
        """Save the current distorted dataset for the given episode index

        Parameters
        ----------

        episode_index: The epsidoe index
        save_index: Flad indicating if the row index should be output as well

        Returns
        -------

        None
        """

        self.distorted_data_set.save_to_csv(
            filename=Path(str(self.config.distorted_set_path) + "_" + str(episode_index)),
            save_index=save_index)

    def create_bins(self) -> None:
        """Create the bins for the state space

        Returns
        -------

        """
        if self.config.env_type == DiscreteEnvType.MULTI_COLUMN_STATE:
            self._create_multi_column_state_bins()
        else:

            self.state_bins = np.linspace(0.0, 1.0, self.config.n_states)

    def get_min_aggregated_state(self) -> Any:
        """Returns the aggregated state for minimum distortions

        Returns
        -------

        """

        if self.config.env_type == DiscreteEnvType.MULTI_COLUMN_STATE:

            state = []
            for name in self.config.min_distortion:
                bin_idx = int(np.digitize(self.config.min_distortion[name], self.column_bins[name]))
                state.append(bin_idx)
            return tuple(state)
        else:
            return int(np.digitize(self.config.min_distortion, self.state_bins))

    def get_max_aggregated_state(self) -> Any:
        """Returns the aggregated state for minimum distortions

        Returns
        -------

        """

        if self.config.env_type == DiscreteEnvType.MULTI_COLUMN_STATE:

            state = []
            for name in self.config.max_distortion:
                bin_idx = int(np.digitize(self.config.max_distortion[name], self.column_bins[name]))
                state.append(bin_idx)
            return tuple(state)
        else:
            return int(np.digitize(self.config.max_distortion, self.state_bins))

    def get_aggregated_state(self, state_val: Any, column_name: str = None) -> Any:
        """Returns the aggregated state given the distortion.
        In the case where only 1D state space is assumed this
        will be the bin index that the state_val corresponds to.
        Else it returns a tuple of bin indices such that
        for every column it indicates the bin index of the distortion
        corresponding to the column

        Parameters
        ----------

        state_val:

        Returns
        -------

        """

        if self.config.state_as_distances:
            if column_name is None:
                column_dists = [self.column_distances[name] for name in self.column_bins] #if
                                #self.config.column_types[name] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE]
            else:
                column_dists = [self.column_distances[name] for name in self.column_bins]

            return column_dists

        if self.config.env_type == DiscreteEnvType.MULTI_COLUMN_STATE:

            if column_name is not None and column_name not in self.column_bins:
                raise ValueError("Name {0} not in column bins names {1} ".format(column_name, list(self.column_bins.keys())))

            if column_name is None:
                column_dists = [(self.column_distances[name], name) for name in self.column_bins if self.config.column_types[name] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE]

            else:
                column_dists = [(self.column_distances[name], name) for name in self.column_bins if self.config.column_types[name] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE]

            state = []
            for distortion, name in column_dists:
                bin_idx = int(np.digitize(distortion, self.column_bins[name]))
                state.append(bin_idx)

            return tuple(state)

        else:

            return int(np.digitize(state_val, self.state_bins))

    def initialize_column_counts(self) -> None:
        """Initialize the column counts

        Returns
        -------

        """

        col_names = self.config.data_set.get_columns_names()
        for name in col_names:
            self.column_visits[name] = 0

            # if it is an identifying attribute
            # we have visited it
            if self.config.column_types[name] == ColumnType.IDENTIFYING_ATTRIBUTE:
                self.column_visits[name] = 1

    def all_columns_visited(self) -> bool:
        """Returns true is all columns have been visited

        Returns
        -------

        """

        return all(self.column_visits.values())

    def initialize_distances(self) -> None:
        """Initialize the structure that holds the
        distances for every column

        Returns
        -------

        None
        """

        col_names = self.config.data_set.get_columns_names()

        if self.config.use_identifying_column_dist_in_total_dist:
            for name in col_names:
                if self.config.column_types[name] == ColumnType.IDENTIFYING_ATTRIBUTE:

                    current_column = self.distorted_data_set.get_column(col_name=name)
                    start_column = self.config.data_set.get_column(col_name=name)
                    datatype = 'float'
                    if self.distorted_data_set.columns[name] == str:

                        current_column = "".join(current_column.values)
                        start_column = "".join(start_column.values)
                        datatype = 'str'
                    distance = self.config.distortion_calculator.calculate(current_column,
                                                                               start_column, datatype)
                    self.column_distances[name] = self.config.use_identifying_column_dist_factor * distance
                else:
                    self.column_distances[name] = 0.0
        else:

            for col in col_names:
                self.column_distances[col] = 0.0

    def apply_action(self, action: ActionBase) -> None:
        """Apply the given action on the underlying data set

        Parameters
        ----------
        action: The action to apply

        Returns
        -------

        None
        """

        # update the column visit count
        self.column_visits[action.column_name] += 1

        # nothing to act on identity
        if action.action_type == ActionType.IDENTITY:
            # the distortion for the column has not changed
            return

        # apply the transform of the data set
        self.distorted_data_set.apply_column_transform(column_name=action.column_name,
                                                       transform=action)

        # what is the previous and current values for the column
        current_column = self.distorted_data_set.get_column(col_name=action.column_name)
        start_column = self.config.data_set.get_column(col_name=action.column_name)

        datatype = 'float'
        # calculate column distortion
        if self.distorted_data_set.columns[action.column_name] == str:
            current_column = "".join(current_column.values)
            start_column = "".join(start_column.values)
            datatype = 'str'

        distance = self.config.distortion_calculator.calculate(current_column,
                                                               start_column, datatype)

        self.column_distances[action.column_name] = distance

    def total_current_distortion(self) -> float:
        """The total distortion in the dataset

        Returns
        -------
        a float representing the total distortion of the dataset
        """

        return self.config.distortion_calculator.total_distortion(list(self.column_distances.values()))

    def reset(self, **options) -> TimeStep:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Parameters
        ----------
        options

        Returns
        -------

        An instance of `TimeStep`
        """

        # reset the copy of the dataset we hold
        self.distorted_data_set = copy.deepcopy(self.config.data_set)
        self._distort_identifying_attributes()
        self.n_rounds_below_min_distortion = 0

        # initialize the  distances for
        # the environment
        self.initialize_distances()
        self.initialize_column_counts()

        # the value of the state on reset is that no
        # distortion of the dataset exists
        state = self.get_aggregated_state(state_val=0.0, column_name=None)

        self.current_time_step = TimeStep(step_type=StepType.FIRST, reward=0.0,
                                          observation=state, discount=self.config.gamma,
                                          info={"total_distortion": 0.0})

        return self.current_time_step

    def step(self, action: ActionBase) -> TimeStep:
        """Step in the environment i.e. apply the action given

        Parameters
        ----------
        action: The action to apply on the environment

        Returns
        -------

        An instance of the TimeStep class
        """

        # apply the action and update distoration
        # and column count

        if isinstance(action, int) or isinstance(action, np.int64):
            action = self.get_action(aidx=action)
        elif isinstance(action, torch.Tensor):
            action = self.get_action(aidx=action.item())

        self.apply_action(action=action)

        column_dist = self.column_distances[action.column_name]

        # calculate the distortion of the dataset
        current_distortion = self.total_current_distortion()

        # the first exit condition. If all columns
        # have been visited then we exit
        done1 = self.all_columns_visited()

        # if the current distortion is greater than
        # the maximum allowed distortion then end the
        # episode; the agent failed. This should be reflected
        # in the received reward
        done2 = current_distortion > self.config.max_total_distortion

        # TODO: We want to consider also the scenario where
        # the same action is chosen over and over again
        done = done1 or done2

        if done is False:
            # if we start (and we do) at distortion 0.0 then
            # this condition may always be true and we do
            # no progress
            done3 = current_distortion < self.config.min_total_distortion

            # if we are trapped below min distortion
            # and we have exceeded the amount of rounds we can stay there
            # then finish the episode
            if done3 and self.n_rounds_below_min_distortion >= self.config.n_rounds_below_min_distortion:
                done = True
            else:
                self.n_rounds_below_min_distortion += 1

        # by default we are at the middle of the episode
        step_type = StepType.MID

        # get the new aggregated state., If the self.config.env_type = DiscreteEnvType.MULTI_COLUMN_STATE
        # then the state_val is not used instead we use the computed distances
        next_state = self.get_aggregated_state(state_val=current_distortion,
                                               column_name=action.column_name)

        if next_state is None:
            raise ValueError("Next state is None")

        # get the bins for the min distortion
        min_dist_bin = self.get_min_aggregated_state()
        max_dist_bin = self.get_max_aggregated_state()

        # get the reward for the current distortion
        # We get reward according to the total dataset distortion
        # or according to how the manager decides
        reward = self.config.reward_manager.get_reward_for_state(total_distortion=current_distortion,
                                                                 current_state=self.current_time_step.observation,
                                                                 next_state=next_state, min_dist_bins=min_dist_bin,
                                                                 **{"action": action, "column_distortion": column_dist})

        # TODO: these modifications will cause the agent to always
        # move close to transition points Also need to account for
        # tuple states
        """
        if next_state is not None and self.current_time_step.observation is not None:
            if next_state < min_dist_bin <= self.current_time_step.observation:
                # the agent chose to step into the chaos again
                # we punish him with double the reward
                reward = self.config.punish_factor * self.config.reward_manager.out_of_min_bound_reward
            elif next_state > max_dist_bin >= self.current_time_step.observation:
                # the agent is going to chaos from above
                # punish him
                reward = self.config.punish_factor * self.config.reward_manager.out_of_max_bound_reward

            elif next_state >= min_dist_bin > self.current_time_step.observation:
                # the agent goes towards the transition of min point so give a higher reward
                # for this
                reward = self.config.reward_factor * self.config.reward_manager.in_bounds_reward

            elif next_state <= max_dist_bin < self.current_time_step.observation:
                # the agent goes towards the transition of max point so give a higher reward
                # for this
                reward = self.config.reward_factor * self.config.reward_manager.in_bounds_reward
        """

        if next_state is None:
            done = True

        if self.config.env_type == DiscreteEnvType.TOTAL_DISTORTION_STATE \
                and next_state >= self.n_states:
            done = True

        if done:
            step_type = StepType.LAST
            #next_state = None

        self.current_time_step = TimeStep(step_type=step_type,
                                          reward=reward,
                                          observation=next_state,
                                          discount=self.config.gamma,
                                          info={"total_distortion": current_distortion})

        return self.current_time_step

    def _create_multi_column_state_bins(self) -> None:

        # create the column bins
        for name in self.column_names:

            # we create bins only for the QUASI_IDENTIFYING_ATTRIBUTE
            # attributes
            if self.config.column_types[name] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE:
                self.column_bins[name] = np.linspace(0.0, 1.0, self.config.n_states)
            #else:
            #    self.column_bins["all_the_rest"] = np.linspace(0.0, 1.0, self.config.n_states)

        if len(self.column_bins) == 3:
            self._build_three_columns()
        elif len(self.column_bins) == 4:
            self._build_4_columns()
        elif len(self.column_bins) == 5:
            self._build_5_columns()
        else:
            raise ValueError("Invalid number of columns. Cannot build the multi-column state bins")

        # add the remaining columns
        for name in self.column_names:
            if self.config.column_types[name] != ColumnType.QUASI_IDENTIFYING_ATTRIBUTE:
                self.column_bins[name] = np.linspace(0.0, 1.0, self.config.n_states)

    def _build_three_columns(self):

        name = ""
        for n in self.config.column_types:
            if self.config.column_types[n] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE:
                name = n
                break

        if name == "":
            raise ValueError("No QUASI_IDENTIFYING_ATTRIBUTE has been specified")

        for i in range(len(self.column_bins[name])):
            for j in range(len(self.column_bins[name])):
                for k in range(len(self.column_bins[name])):
                    self.state_space.append((i, j, k))

    def _build_4_columns(self):

        name = ""
        for n in self.config.column_types:
            if self.config.column_types[n] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE:
                name = n
                break

        if name == "":
            raise ValueError("No QUASI_IDENTIFYING_ATTRIBUTE has been specified")

        for i1 in range(len(self.column_bins[name])):
            for i2 in range(len(self.column_bins[name])):
                for i3 in range(len(self.column_bins[name])):
                    for i4 in range(len(self.column_bins[name])):
                        self.state_space.append((i1, i2, i3, i4))

    def _build_5_columns(self):

        name = ""
        for n in self.config.column_types:
            if self.config.column_types[n] == ColumnType.QUASI_IDENTIFYING_ATTRIBUTE:
                name = n
                break

        if name == "":
            raise ValueError("No QUASI_IDENTIFYING_ATTRIBUTE has been specified")

        name = self.column_names[0]
        for i1 in range(len(self.column_bins[name])):
            for i2 in range(len(self.column_bins[name])):
                for i3 in range(len(self.column_bins[name])):
                    for i4 in range(len(self.column_bins[name])):
                        for i5 in range(len(self.column_bins[name])):
                            self.state_space.append((i1, i2, i3, i4, i5))

    def _distort_identifying_attributes(self):

        for name in self.config.column_types:

            if self.config.column_types[name] == ColumnType.IDENTIFYING_ATTRIBUTE:
                # we need to alter the column
                action = ActionTransform(column_name=name, transform_value='*')
                self.distorted_data_set.apply_column_transform(column_name=name,
                                                               transform=action)


