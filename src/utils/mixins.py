"""
Various mixin classes to use for simplifying  code
"""

import numpy as np
from typing import TypeVar, Any

QTable = TypeVar('QTable')
Hierarchy = TypeVar('Hierarchy')


class WithHierarchyTable(object):

    def __init__(self) -> None:
        self.table = {}
        self.iterators = []

    def add_hierarchy(self, key: str, hierarchy: Hierarchy) -> None:
        """
        Add a hierarchy for the given key
        :param key: The key to attach the Hierarchy
        :param hierarchy: The hierarchy to attach
        :return: None
        """
        self.table[key] = hierarchy

    def reset_iterators(self):
        """
            Reinitialize the iterators in the table
            :return:
        """

        # fill in the iterators
        self.iterators = [iter(self.table[item]) for item in self.table]

    def finished(self) -> bool:
        """
        Returns true if the action has exhausted all its
        transforms
        :return:
        """
        exhausted = True

        for item in self.table:
            if not self.table[item].is_exhausted():
                return False

        return exhausted


class WithQTableMixin(object):
    """
    Helper class to associate a q_table with an algorithm
     if this is needed.
    """
    def __init__(self):
        # the table representing the q function
        # client code should choose the type of
        # the table
        self.q_table: QTable = None


class WithMaxActionMixin(object):
    """
    The class WithMaxActionMixin.
    """

    def __init__(self):
        super(WithMaxActionMixin, self).__init__()
        self.q_table: QTable = None

    def max_action(self, state: Any, n_actions: int) -> int:
        """
        Return the action index that presents the maximum
        value at the given state
        :param state: state index
        :param n_actions: Total number of actions allowed
        :return: The action that corresponds to the maximum value
        """
        values = np.array(self.q_table[state, a] for a in range(n_actions))
        action = np.argmax(values)
        return int(action)
