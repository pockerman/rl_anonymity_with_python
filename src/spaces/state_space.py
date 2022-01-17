"""
Discretized state space
"""

from typing import TypeVar, List
from gym.spaces.discrete import Discrete

from src.exceptions.exceptions import Error

ActionStatus = TypeVar("ActionStatus")
Env = TypeVar("Env")


class State(object):
    """
    Describes an environment state
    """
    def __init__(self, column_name: str, state_id: int):
        self.column_name: str = column_name
        self.state_id: int = state_id
        self.history: List[ActionStatus] = []

    @property
    def key(self) -> tuple:
        return self.column_name, self.state_id


class StateSpace(Discrete):
    """
    The StateSpace class. Models a discrete state space
    """

    def __init__(self) -> None:
        super(StateSpace, self).__init__(n=0)
        self.states = {}

    def __iter__(self):
        """
        Make the class Iterable. We need to override __iter__() function inside our class.
        :return:
        """
        return self.states.__iter__()

    def __getitem__(self, item) -> State:
        return self.states[item]

    def init_from_environment(self, env: Env):
        """
        Initialize from environment
        :param env:
        :return:
        """
        names = env.feature_names
        for col_name in names:

            if col_name in self.states:
                raise ValueError("Column {0} already exists".format(col_name))

            self.states[col_name] = State(column_name=col_name, state_id=len(self.states))

        # set the number of discrete states
        self.n = len(self.states)

    def add_state(self, state: State) -> None:
        """
        Add a new state in the state space
        :param state:
        :return:
        """
        if state.column_name in self.states:
            raise ValueError("Column {0} already exists".format(state.column_name))

        self.states[state.column_name] = state

    def update_state(self, state_name, status: ActionStatus):
        self.states[state_name].history.append(status)

    def get_state_by_name(self, name) -> State:
        return self.states[name]

    def __len__(self):
        """
        Returns the size of the state space
        :return:
        """
        return len(self.states)
