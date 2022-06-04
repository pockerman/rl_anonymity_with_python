"""module serial_hierarchy. A SerialHierarchy represents a hierarchy of transformations
that are applied one after the other

"""

from typing import List, Any


class SerialHierarchy(object):

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
        self.hierarchy: dict = values

    def __getitem__(self, item):
        """
        Returns the item-th item
        :param item:
        :return:
        """
        return self.hierarchy[item]

    def __setitem__(self, key, value):
        """
        Set the key-th item to the given value.
        If the key-th item has already been set it overrides
        the existing value
        :param key:
        :param value:
        :return:
        """
        self.hierarchy[key] = value

    def add(self, key: Any, values: Any) -> None:
        self.hierarchy[key] = values

    def __len__(self):
        """
        Returns the size of the hierarchy
        :return:
        """
        return len(self.hierarchy)

