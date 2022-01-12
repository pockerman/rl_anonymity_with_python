"""
RL Environment API taken from
https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""

import copy
import enum
import numpy as np
import pandas as pd
import torch
from typing import NamedTuple, Generic, Optional, TypeVar
import multiprocessing as mp

from src.exceptions.exceptions import Error
from src.spaces.actions import ActionBase, ActionType
from src.spaces.state_space import StateSpace, State
from src.utils.string_distance_calculator import DistanceType, TextDistanceCalculator

DataSet = TypeVar("DataSet")
RewardManager = TypeVar("RewardManager")
ActionSpace = TypeVar("ActionSpace")

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
    reward: Optional[_Reward]
    discount: Optional[_Discount]
    observation: _Observation

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST


class EnvConfig(object):
    """
    The EnvConfig class. Wrapper for the various parameters
    passed in the Environment class constructor
    """
    def __init__(self):
        self.data_set: DataSet = None
        self.action_space: ActionSpace = None
        self.reward_manager: RewardManager = None
        self.average_distortion_constraint: float = 0
        self.start_column: str = "None_Column"
        self.gamma: float = 0.99


class Environment(object):
    """
    The Environment class. Wrapper to a data set to act
    as an environment suitable for reinforcement learning
    """

    def __init__(self, env_config: EnvConfig):
        self.data_set = env_config.data_set
        self.start_ds = copy.deepcopy(env_config.data_set)
        self.current_time_step = self.start_ds
        self.action_space = env_config.action_space
        self.gamma = env_config.gamma
        self.start_column = env_config.start_column
        self.current_column: str = env_config.start_column
        self.columns: list = self.data_set.get_columns_names()
        self.current_column_idx = 0
        self.column_distances = {}
        self.state_space = StateSpace()
        self.distance_calculator = None
        self.reward_manager: RewardManager = env_config.reward_manager

        # initialize the state space
        self.state_space.init_from_environment(env=self)

    @property
    def observation_space(self) -> StateSpace:
        """
        Returns the state space
        :return:
        """
        return self.state_space

    @property
    def n_features(self) -> int:
        """
        Returns the number of features in the dataset
        :return:
        """
        return self.start_ds.n_columns

    @property
    def feature_names(self) -> list:
        """
        Returns the feature names in the dataset
        :return:
        """
        return self.start_ds.get_columns_names()

    @property
    def n_examples(self) -> int:
        """
        Returns the number of examples in the data set.
        For a tabular data set this will be the number of rows
        :return:
        """
        return self.start_ds.n_rows

    def initialize_text_distances(self, distance_type: DistanceType) -> None:
        """
        Initialize the text distances for features of type string
        :return: None
        """

        self.distance_calculator = TextDistanceCalculator(dist_type=distance_type)
        col_names = self.start_ds.get_columns_names()
        for col in col_names:
            # check here: https://stackoverflow.com/questions/43652161/numpy-typeerror-data-type-string-not-understood/43652230
            if self.start_ds.columns[col] == str:
                self.column_distances[col] = 0.0 #np.zeros(len(self.start_ds.get_column(col_name=col)))

    def sample_action(self) -> ActionBase:
        """
        Sample an action from the action space
        :return: the sampled action
        """
        return self.action_space.sample_and_get()

    def get_action(self, aidx: int) -> ActionBase:
        """
        Returns the aidx-th action from the action space
        :param aidx:
        :return:
        """
        return self.action_space[aidx]

    def get_column_as_tensor(self, column_name) -> torch.Tensor:
        """
        Returns the column in the dataset as a torch tensor
        :param column_name:
        :return:
        """
        data = {}

        if self.start_ds.columns[column_name] == str:

            numpy_vals = self.column_distances[column_name]
            data[column_name] = numpy_vals
        else:
            data[column_name] = self.data_set.get_column(col_name=column_name).to_numpy()

        target_df = pd.DataFrame(data)
        return torch.tensor(target_df.to_numpy(), dtype=torch.float64)

    def get_ds_as_tensor(self) -> torch.Tensor:

        """
        Returns the underlying data set as a numeric torch Tensor
        :return: torch.Tensor
        """

        col_names = self.start_ds.get_columns_names()
        data = {}
        for col in col_names:

            if self.start_ds.columns[col] == str:
                numpy_vals = self.column_distances[col]
                data[col] = numpy_vals
            else:
                data[col] = self.data_set.get_column(col_name=col).to_numpy()

        target_df = pd.DataFrame(data)
        return torch.tensor(target_df.to_numpy(), dtype=torch.float64)

    def prepare_column_state(self, column_name):
        """
        Prepare the column state to a numeric value
        :param column_name:
        :return:
        """
        if self.start_ds.columns[column_name] == str:

            if self.distance_calculator is None:
                raise Error("Distance calculator is not set. "
                            "Have you called self.initialize_text_distances?")

            # what is the previous and current values for the column
            current_column = self.data_set.get_column(col_name=column_name)
            start_column = self.start_ds.get_column(col_name=column_name)

            row_count = 0
            print("Distance {0} ".format(self.distance_calculator.calculate(txt1="".join(current_column.values),
                                                                            txt2="".join(start_column.values))))

            self.column_distances[column_name] = self.distance_calculator.calculate(txt1="".join(current_column.values),
                                                                                    txt2="".join(start_column.values))
            #for item1, item2 in zip(current_column.values, start_column.values):
            #    #self.column_distances[column_name][row_count] = self.distance_calculator.calculate(txt1=item1, txt2=item2)

            #    row_count += 1

    def prepare_columns_state(self):
        """
        Prepare the column states to be sent to the agent.
        If a column is a string we calculate the  cosine distance
        :return:
        """

        if self.distance_calculator is None:
            raise Error("Distance calculator is not set. "
                        "Have you called self.initialize_text_distances?")

        col_names = self.data_set.get_columns_names()
        for col in col_names:
            # check here: https://stackoverflow.com/questions/43652161/numpy-typeerror-data-type-string-not-understood/43652230
            #if self.data_set.get_column_type(col_name=col) == np.dtype('str'):
            if self.start_ds.columns[col] == str:

                # what is the previous and current values for the column
                current_column = self.data_set.get_column(col_name=col)
                start_column = self.start_ds.get_column(col_name=col)

                for item1, item2 in zip(current_column.values, start_column.values):

                    self.column_distances[col] = self.distance_calculator.calculate(txt1=item1, txt2=item2)

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

        self.current_column_idx = 0
        self.current_column = self.start_column #self.columns[self.current_column_idx]

        # reset the action space so that we can
        # re-apply any transformations
        self.action_space.reset()

        # initialize the text distances for
        # the environment
        self.initialize_text_distances(distance_type=self.distance_calculator.distance_type)

        # get the DS as a torch tensor
        #observation = self.start_ds.get_column(col_name=self.start_column)

        state = self.state_space.get_state_by_name(name=self.start_column)

        self.current_time_step = TimeStep(step_type=StepType.FIRST, reward=0.0,
                                          observation=state, discount=self.gamma)

        # update internal data
        self.current_column_idx += 1
        self.current_column = self.columns[self.current_column_idx]
        return self.current_time_step

    def apply_action(self, action: ActionBase):
        """
        Apply the action on the environment
        :param action: The action to apply on the environment
        :return:
        """

        if action.action_type == ActionType.IDENTITY:
            return

        # apply the transform of the data set
        self.data_set.apply_column_transform(column_name=action.column_name, transform=action)

    def step(self, action: ActionBase) -> TimeStep:
        """

        Updates the environment according to the action and returns a `TimeStep`.
        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and `action`
        will be ignored.

        This method will also start a new sequence if called after the environment
        has been constructed and `reset` has not been called. Again, in this case
        `action` will be ignored.
        """

        print("Applying action {0} on column {1}".format(action.action_type.name, action.column_name))

        if action.is_exhausted():
            # the selected action is exhausted
            # by choosing such an action gives neither good
            # or bad?
            return TimeStep(step_type=StepType.LAST, reward=0.0,
                            observation=None, discount=self.gamma)

        # apply the action
        self.apply_action(action=action)

        # update the state space
        self.state_space.update_state(state_name=action.column_name, status=action.action_type)

        self.prepare_column_state(column_name=action.column_name)

        # perform the action on the data set
        #self.prepare_columns_state()

        # calculate the information leakage and establish the reward
        # to return to the agent
        reward = self.reward_manager.get_state_reward(self.state_space, action)

        # what is the next state? maybe do it randomly?
        # or select the next column in the dataset
        self.current_column_idx += 1

        # check if the environment is finished
        if self.current_column_idx >= len(self.columns):
            return TimeStep(step_type=StepType.LAST, reward=0.0,
                            observation=None, discount=self.gamma)

        if self.action_space.is_exhausted():
            return TimeStep(step_type=StepType.LAST, reward=0.0,
                            observation=None, discount=self.gamma)

        self.current_column = self.columns[self.current_column_idx]
        next_state = self.state_space.get_state_by_name(name=self.current_column)

        return TimeStep(step_type=StepType.MID, reward=reward,
                        observation=next_state, #self.get_column_as_tensor(column_name=action.column_name).float(),
                        discount=self.gamma)


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
