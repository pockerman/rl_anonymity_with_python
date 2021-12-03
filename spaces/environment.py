"""
RL Environment API taken from
https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""

import abc
import copy
import enum
import numpy as np
import pandas as pd
import torch
from typing import Any, NamedTuple, Generic, Optional, TypeVar
import multiprocessing as mp

from exceptions.exceptions import Error
from spaces.actions import ActionBase
from utils.string_sequence_calculator import DistanceType, TextDistanceCalculator

DataSet = TypeVar("DataSet")

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


class Environment(object):

    def __init__(self, data_set, action_space,
                 gamma: float, start_column: str, ):
        self.data_set = data_set
        self.start_ds = copy.deepcopy(data_set)
        self.current_time_step = self.start_ds
        self.action_space = action_space
        self.gamma = gamma
        self.start_column = start_column
        self.column_distances = {}
        self.distance_calculator = None

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
                self.column_distances[col] = np.zeros(len(self.start_ds.get_column(col_name=col)))

    def sample_action(self):
        return self.action_space.sample_and_get()

    def get_numeric_ds(self) -> torch.Tensor:

        """
        Returns the underlying data set as a numeric torch Tensor
        :return: torch.Tensor
        """

        col_names = self.start_ds.get_columns_names()
        data = {}
        for col in col_names:

            if self.start_ds.columns[col] == str:
                print("col: {0} type {1}".format(col, self.start_ds.get_column_type(col_name=col)))
            #if self.start_ds.get_column_type(col_name=col) == np.dtype('str'):
                numpy_vals = self.column_distances[col]
                data[col] = numpy_vals
            else:
                data[col] = self.data_set.get_column(col_name=col).to_numpy()

        target_df = pd.DataFrame(data)
        return torch.tensor(target_df.to_numpy())

    def prepare_column_states(self):
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
            if self.data_set.get_column_type(col_name=col) == np.dtype('str'):

                # what is the previous and current values for the column
                current_column = self.data_set.get_column(col_name=col)
                start_column = self.start_ds.get_column(col_name=col)

                for item1, item2 in zip(current_column.vaslues, start_column.values):

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

        # initialize the text distances for
        # the environment
        self.initialize_text_distances(distance_type=self.distance_calculator.distance_type)

        # get the DS as a torch tensor

        observation = self.start_ds.get_column(col_name=self.start_column)
        self.current_time_step = TimeStep(step_type=StepType.FIRST, reward=0.0,
                                          observation=self.start_ds, discount=self.gamma)
        return self.current_time_step

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

        # perform the action on the data set

        return self.current_time_step


def get_ds_as_torch_tensor(ds: Environment) -> torch.Tensor:
    pass




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
