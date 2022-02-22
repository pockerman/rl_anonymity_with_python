"""
Discretized state space
"""

from typing import TypeVar, List, Any
from src.exceptions.exceptions import Error

ActionStatus = TypeVar("ActionStatus")
Env = TypeVar("Env")


class StateIterator(object):
    """
    StateIterator class. Helper class to iterate over the
    columns of  a State object
    """

    def __init__(self, values: List):
        self.current_position = 0
        self.values = values

    @property
    def at(self) -> Any:
        """
        Returns the value of the iterator at the current position
        without incrementing the position of the iterator
        :return: Any
        """
        return self.values[self.current_position]

    @property
    def finished(self) -> bool:
        """
        Returns true if the iterator is exhausted
        :return:
        """
        return self.current_position >= len(self.values)

    def __next__(self):

        if self.current_position < len(self.values):
            result = self.values[self.current_position]
            self.current_position += 1
            return result

        raise StopIteration

    def __len__(self):
        """
        Returns the total number of items in the iterator
        :return:
        """
        return len(self.values)


class State(object):
    """
    Helper to represent a State
    """
    def __init__(self):
        self.idx = -1
        self.bin_idx = -1
        self.total_distortion: float = 0.0
        self.column_distortions = {}

    def __contains__(self, item) -> bool:
        return item in self.column_distortions.keys()

    def __iter__(self):
        return StateIterator(list(self.column_distortions.keys()))

    def __getitem__(self, name: str) -> float:
        """
        Get the distortion corresponding to the name-th column

        Parameters
        ----------
        name: The name of the column

        Returns
        -------

        The column distortion
        """
        return self.column_distortions[name]


