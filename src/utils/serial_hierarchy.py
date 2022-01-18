"""
A SerialHierarchy represents a hierarchy of transformations
that are applied one after the other
"""

from typing import List, Any
from src.utils.hierarchy_base import HierarchyBase
from src.utils.updateable_map import UpdateableMap


class SerialtHierarchyIterator(object):
    """
    SerialtHierarchyIterator class. Helper class to iterate over a
    SerialHierarchy object
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


class SerialHierarchy(HierarchyBase):

    """
    A SerialHierarchy represents a hierarchy of transformations
    that are applied one after the other. Applications should explicitly
    provide the list of the ensuing transformations. For example assume that the
    data field has the value 'foo' then values
     the following list ['fo*', 'f**', '***']
    """
    def __init__(self, values: dict) -> None:
        """
        Constructor. Initialize the hierarchy by passing the
        list of the ensuing transformations.
        :param values:
        """
        super(SerialHierarchy, self).__init__()
        self.hierarchy: dict = values

    def __getitem__(self, item):
        """
        Returns the item-th item
        :param item:
        :return:
        """
        return self.hierarchy[item]

    def add(self, key: Any, values: Any) -> None:
        self.hierarchy[key] = values

    def __len__(self):
        """
        Returns the size of the hierarchy
        :return:
        """
        return len(self.hierarchy)

