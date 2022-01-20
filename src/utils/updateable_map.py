"""
Wrapper class over the traditional dict class.
Every time the get_value function is called,
the key is updated with the current value that
the counter points to
"""
from typing import Any, List, KeysView, ValuesView
from src.exceptions.exceptions import InvalidStateException


class UpdateableMap(object):

    def __init__(self, list_size: int) -> None:
        self.current_pos = {}
        self._values = {}
        self.list_size: int = list_size

    def __getitem__(self, item) -> Any:
        return self._values[item]

    def __len__(self):
        return len(self._values)

    def keys(self) -> KeysView:
        """
        Returns the current keys
        :return:
        """
        return self._values.keys()

    def values(self) -> ValuesView:
        """
        Returns the current values in the map
        :return:
        """
        return self._values.values()

    def get_value(self, key: Any) -> Any:

        values = self._values[key]
        current_pos = self.current_pos[key]

        val = values[current_pos]

        # update keys
        self._values[val] = self._values[key]
        self._values[val][current_pos] = key
        del self._values[key]

        current_pos += 1
        self.current_pos[val] = current_pos
        return val

    def insert(self, key: Any, values: List[Any]) -> None:

        if len(values) != self.list_size:
            raise ValueError("Invalid list size. Size={0} should be equal to {1}".format(len(values), self.list_size))

        self._values[key] = values
        self.current_pos[key] = 0

    def is_exhausted(self) -> bool:
        """
        Return true if all keys are self.tail_value
        :return:
        """
        counters = self.current_pos.values()
        return all(counter >= self.list_size for counter in counters)
