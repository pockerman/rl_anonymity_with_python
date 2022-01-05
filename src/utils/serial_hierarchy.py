"""
A SerialHierarchy represents a hierarchy of transformations
that are applied one after the other
"""

from typing import List, Any
from src.utils.hierarchy_base import HierarchyBase


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

    def __next__(self):

        if self.current_position < len(self.values):
            result = self.values[self.current_position]
            self.current_position += 1
            return result

        raise StopIteration


class SerialHierarchy(HierarchyBase):

    """
    A SerialHierarchy represents a hierarchy of transformations
    that are applied one after the other. Applications should explicitly
    provide the list of the ensuing transformations. For example assume that the
    data field has the value 'foo' then values
the following list ['fo*', 'f**', '***']
    """
    def __init__(self, values: List) -> None:
        """
        Constructor. Initialize the hierarchy by passing the
        list of the ensuing transformations.
        :param values:
        """
        super(SerialHierarchy, self).__init__()
        self.iterator = SerialtHierarchyIterator(values=values)

    def __iter__(self):
        """
        Make the class Iterable. We need to override __iter__() function inside our class.
        :return:
        """
        return self.iterator

    @property
    def value(self) -> Any:
        """
        :return: the current value the hierarchy assumes
        """
        return self.iterator.at
