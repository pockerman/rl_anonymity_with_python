"""
RL Environment API taken from
https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""

import copy
import enum
import numpy as np
from pathlib import Path
import pandas as pd
import torch
from typing import NamedTuple, Generic, Optional, TypeVar, List
import multiprocessing as mp

from src.spaces.actions import ActionBase, ActionType
from src.utils.string_distance_calculator import StringDistanceType, TextDistanceCalculator
from src.utils.numeric_distance_type import NumericDistanceType
from src.utils.numeric_distance_calculator import NumericDistanceCalculator

DataSet = TypeVar("DataSet")
RewardManager = TypeVar("RewardManager")
ActionSpace = TypeVar("ActionSpace")
DistortionCalculator = TypeVar('DistortionCalculator')

_Reward = TypeVar('_Reward')
_Discount = TypeVar('_Discount')
_Observation = TypeVar('_Observation')


class StepType(enum.IntEnum):
    """
      Defines the status of a `TimeStep` within a sequence.
      """

    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0

    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1

    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST


class TimeStep(NamedTuple, Generic[_Reward, _Discount, _Observation]):
    step_type: StepType
    info: dict
    reward: Optional[_Reward]
    discount: Optional[_Discount]
    observation: _Observation

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST


class DiscreteEnvConfig(object):
    """
    Configuration for discrete environment
    """

    def __init__(self) -> None:
        self.data_set: DataSet = None
        self.action_space: ActionSpace = None
        self.reward_manager: RewardManager = None
        self.average_distortion_constraint: float = 0.0
        self.gamma: float = 0.99
        # self.string_column_distortion_type: StringDistanceType = StringDistanceType.INVALID
        # self.numeric_column_distortion_metric_type: NumericDistanceType = NumericDistanceType.INVALID
        self.n_states: int = 10
        self.min_distortion: float = 0.4
        self.max_distortion: float = 0.7
        self.n_rounds_below_min_distortion: int = 10
        self.distorted_set_path: Path = None
        self.distortion_calculator: DistortionCalculator = None


class DiscreteStateEnvironment(object):
    """
    The DiscreteStateEnvironment class. Uses state aggregation in order
    to create bins where the average total distortion of the dataset falls in
    """

    def __init__(self, env_config: DiscreteEnvConfig) -> None:
        self.config = env_config
        self.n_rounds_below_min_distortion = 0
        self.state_bins: List[float] = []
        self.distorted_data_set = copy.deepcopy(self.config.data_set)
        self.current_time_step: TimeStep = None
        # self.string_distance_calculator: TextDistanceCalculator = None

        # dictionary that holds the distortion for every column
        # in the dataset
        self.column_distances = {}

        # hold a copy of the visits per
        # column. An episode ends when all columns
        # have been visited
        self.column_visits = {}
        self.create_bins()

    @property
    def action_space(self):
        return self.config.action_space

    @property
    def n_actions(self) -> int:
        return len(self.config.action_space)

    @property
    def n_states(self) -> int:
        return self.config.n_states

    def get_action(self, aidx: int) -> ActionBase:
        return self.config.action_space[aidx]

    def save_current_dataset(self, episode_index: int, save_index: bool = False) -> None:
        """
        Save the current distorted datase for the given episode index
        :param episode_index:
        :param save_index:
        :return:
        """
        self.distorted_data_set.save_to_csv(
            filename=Path(str(self.config.distorted_set_path) + "_" + str(episode_index)),
            save_index=save_index)

    def create_bins(self) -> None:
        """
        Create the bins
        :return:
        """
        self.state_bins = np.linspace(0.0, 1.0, self.config.n_states)

    def get_aggregated_state(self, state_val: float) -> int:
        """
        Returns the bin index that the state_val corresponds to
        :param state_val: The value of the state. This typically will be
        either a column normalized distortion value or the dataset average total
        distortion
        :return:
        """
        return int(np.digitize(state_val, self.state_bins))

    def initialize_column_counts(self) -> None:
        """
        Set the column visit counts to zero
        :return:
        """
        col_names = self.config.data_set.get_columns_names()
        for col in col_names:
            self.column_visits[col] = 0

    def all_columns_visited(self) -> bool:
        """
        Returns True is all column counts are greater than zero
        :return:
        """
        return all(self.column_visits.values())

    def initialize_distances(self) -> None:
        """
        Initialize the text distances for features of type string. We set the
        normalized distance to 0.0 meaning that no distortion is assumed initially
        :return: None
        """
        # self.string_distance_calculator = TextDistanceCalculator(dist_type=self.config.string_column_distortion_type)
        col_names = self.config.data_set.get_columns_names()
        for col in col_names:
            self.column_distances[col] = 0.0

    def apply_action(self, action: ActionBase):
        """
        Apply the action on the environment
        :param action: The action to apply on the environment
        :return:
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
        """
        Calculates the current total distortion of the dataset.
        :return:
        """

        return self.config.distortion_calculator.total_distortion(
            list(self.column_distances.values()))

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

        # reset the copy of the dataset we hold
        self.distorted_data_set = copy.deepcopy(self.config.data_set)
        self.n_rounds_below_min_distortion = 0

        # initialize the  distances for
        # the environment
        self.initialize_distances()
        self.initialize_column_counts()

        # the value of the state on reset is that no
        # distortion of the dataset exists
        state = self.get_aggregated_state(state_val=0.0)

        self.current_time_step = TimeStep(step_type=StepType.FIRST, reward=0.0,
                                          observation=state, discount=self.config.gamma,
                                          info={"total_distortion": 0.0})

        return self.current_time_step

    def step(self, action: ActionBase) -> TimeStep:
        """
        Apply the action and return new state
        :param action: The action to apply
        :return:
        """
        # apply the action and update distoration
        # and column count
        self.apply_action(action=action)

        # calculate the distortion of the dataset
        current_distortion = self.total_current_distortion()

        # get the reward for the current distortion
        reward = self.config.reward_manager.get_reward_for_state(state=current_distortion, **{"action": action})

        # the first exit condition
        done1 = self.all_columns_visited()

        # if the current distortion is greater than
        # the maximum allowed distortion then end the
        # episode; the agent failed. This should be reflected
        # in the received reward
        done2 = current_distortion > self.config.max_distortion

        # TODO: We want to consider also the scenario where
        # the same action is chosen over and over again
        done = done1 or done2

        if done is False:
            # if we start (and we do) at distortion 0.0 then
            # this condition may always be true and we do
            # no progress
            done3 = current_distortion < self.config.min_distortion

            # if we are trapped below min distortion
            # and we have exceeded the amount of rounds we can stay there
            # then finish the episode
            if done3 and self.n_rounds_below_min_distortion >= self.config.n_rounds_below_min_distortion:
                done = True
            else:
                self.n_rounds_below_min_distortion += 1

        # by default we are at the middle of the episode
        step_type = StepType.MID
        next_state = self.get_aggregated_state(state_val=current_distortion)

        # get the bin for the min distortion
        min_dist_bin = self.get_aggregated_state(state_val=self.config.min_distortion)
        max_dist_bin = self.get_aggregated_state(state_val=self.config.max_distortion)

        # TODO: these modifications will cause the agent to always
        # move close to transition points
        # TODO: Remove the magic constants
        if next_state is not None and self.current_time_step.observation is not None:
            if next_state < min_dist_bin <= self.current_time_step.observation:
                # the agent chose to step into the chaos again
                # we punish him with double the reward
                reward = 2.0 * self.config.reward_manager.out_of_min_bound_reward
            elif next_state > max_dist_bin >= self.current_time_step.observation:
                # the agent is going to chaos from above
                # punish him
                reward = 2.0 * self.config.reward_manager.out_of_max_bound_reward

            elif next_state >= min_dist_bin > self.current_time_step.observation:
                # the agent goes towards the transition of min point so give a higher reward
                # for this
                reward = 0.95 * self.config.reward_manager.in_bounds_reward

            elif next_state <= max_dist_bin < self.current_time_step.observation:
                # the agent goes towards the transition of max point so give a higher reward
                # for this
                reward = 0.95 * self.config.reward_manager.in_bounds_reward

        if next_state is None or next_state >= self.n_states:
            done = True

        if done:
            step_type = StepType.LAST
            next_state = None

        self.current_time_step = TimeStep(step_type=step_type,
                                          reward=reward,
                                          observation=next_state,
                                          discount=self.config.gamma,
                                          info={"total_distortion": current_distortion})

        return self.current_time_step


class MultiprocessEnv(object):

    def __init__(self, make_env_fn, make_env_kargs, seed, n_workers):
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.n_workers = n_workers
        self.pipes = [mp.Pipe() for rank in range(self.n_workers)]
        self.workers = [
            mp.Process(target=self.work,
                       args=(rank, self.pipes[rank][1])) for rank in range(self.n_workers)]
        [w.start() for w in self.workers]

    def work(self, rank, worker_end):
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed + rank)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                worker_end.send(env.step(**kwargs))
            elif cmd == '_past_limit':
                # Another way to check time limit truncation
                worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                env.close(**kwargs)
                del env
                worker_end.close()
                break

    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(('step', {'action': actions[rank]}), rank) \
         for rank in range(self.n_workers)]
        results = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            o, r, d, _ = parent_end.recv()
            if d:
                self.send_msg(('reset', {}), rank)
                o = parent_end.recv()
            results.append((o,
                            np.array(r, dtype=np.float),
                            np.array(d, dtype=np.float), _))
        return [np.vstack(block) for block in np.array(results).T]
