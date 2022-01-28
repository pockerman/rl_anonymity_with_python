"""
Various mixin classes to use for simplifying  code
"""

import numpy as np
import abc
from typing import TypeVar, Any

from src.exceptions.exceptions import InvalidParamValue

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


class WithQTableMixinBase(metaclass=abc.ABCMeta):
    """
    Base class to impose the concept of Q-table
    """

    def __init__(self):
        # the table representing the q function
        # client code should choose the type of
        # the table
        self.q_table: QTable = None


class WithQTableMixin(WithQTableMixinBase):
    """
    Helper class to associate a q_table with an algorithm
     if this is needed.
    """
    def __init__(self):
        super(WithQTableMixin, self).__init__()

    def state_action_values(self, state: Any, n_actions: int):

        if self.q_table is None:
            raise InvalidParamValue(param_name="q_table", param_value="None")

        values = [self.q_table[state, a] for a in range(n_actions)]
        return values


class WithMaxActionMixin(WithQTableMixin):
    """
    The class WithMaxActionMixin.
    """

    def __init__(self):
        super(WithMaxActionMixin, self).__init__()

    def max_action(self, state: Any, n_actions: int) -> int:
        """
        Return the action index that presents the maximum
        value at the given state
        :param state: state index
        :param n_actions: Total number of actions allowed
        :return: The action that corresponds to the maximum value
        """
        values = self.state_action_values(state, n_actions) #[self.q_table[state, a] for a in range(n_actions)]
        values = np.array(values)
        action = np.argmax(values)
        return int(action)
