from typing import List, Any
from src.utils.hierarchy_base import HierarchyBase


class DefaultHierarchyIterator(object):

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

    def __next__(self):

        if self.current_position < len(self.values):
            result = self.values[self.current_position]
            self.current_position += 1
            return result

        raise StopIteration


class DefaultHierarchy(HierarchyBase):

    def __init__(self, values: List) -> None:
        super(DefaultHierarchy, self).__init__()
        self.iterator = DefaultHierarchyIterator(values=values)

    def __iter__(self):
        """
        Make the class Iterable. We need to override __iter__() function inside our class.
        :return:
        """
        return self.iterator

    @property
    def value(self) -> Any:
        return self.iterator.at

