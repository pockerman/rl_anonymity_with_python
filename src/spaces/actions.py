# -*- coding: utf-8 -*-
"""The actions module. This module includes
various actions to be applied by the implemented RL agents

"""

import abc
import enum
from typing import List, TypeVar, Any
import numpy as np

from src.utils.mixins import WithHierarchyTable

Hierarchy = TypeVar("Hierarchy")


class ActionType(enum.IntEnum):
    """Defines the type of an Action
    """

    INVALID_TYPE = -1
    TRANSFORM = 0
    SUPPRESS = 1
    GENERALIZE = 2
    IDENTITY = 3
    RESTORE = 4

    def invalid(self) -> bool:
        return self is ActionType.RESTORE

    def transform(self) -> bool:
        return self is ActionType.TRANSFORM

    def suppress(self) -> bool:
        return self is ActionType.SUPPRESS

    def generalize(self) -> bool:
        return self is ActionType.GENERALIZE

    def identity(self) -> bool:
        return self is ActionType.IDENTITY

    def restore(self) -> bool:
        return self is ActionType.RESTORE


class ActionBase(metaclass=abc.ABCMeta):
    """Base class for actions
    """

    def __init__(self, column_name: str, action_type: ActionType) -> None:
        """Constructor

        Parameters
        ----------
        column_name : The name of the column this is acting on
        action_type : The type of the action

        """
        self.column_name = column_name
        self.action_type = action_type
        self.idx = None
        self.key = (self.column_name, self.action_type)

    @abc.abstractmethod
    def act(self, **ops) -> Any:
        """Perform the action

        Parameters
        ----------
        ops : The data to distort

        Returns
        -------
        Typically the action returns the distorted subset of the data

        """


class ActionIdentity(ActionBase):
    """Implements the identity action. Use this action
    to signal that no distortion should be applied.

    """

    def __init__(self, column_name: str) -> None:
        """Constructor

        Parameters
        ----------
        column_name : The name of the column this is acting on

        """
        super(ActionIdentity, self).__init__(column_name=column_name, action_type=ActionType.IDENTITY)
        self.called = False

    def act(self, **ops) -> Any:
        """Perform the action

        Parameters
        ----------
        ops : The data to distort

        Returns
        -------

        The distorted column

        """

        self.called = True
        return ops['data']


class ActionRestore(ActionBase, WithHierarchyTable):
    """
    Implements the restore action
    """

    def __init__(self, column_name: str, restore_values: Hierarchy):
        super(ActionRestore, self).__init__(column_name=column_name, action_type=ActionType.RESTORE)
        self.table = restore_values

    def act(self, **ops) -> Any:
        """
        Perform an action
        :return:
        """
        # get the values of the column
        col_vals = ops['data']

        assert len(col_vals) == len(self.table), "Invalid size. Column size does not match self.table size"

        # generalize the data given
        for i, item in enumerate(ops["data"]):
            value = self.table[i]
            col_vals[i] = value

        ops["data"] = col_vals
        return ops['data']


class ActionTransform(ActionBase):
    """ Implements the transform action
    """
    def __init__(self, column_name: str, transform_value: Any):
        super(ActionTransform, self).__init__(column_name=column_name, action_type=ActionType.TRANSFORM)
        self.transform = transform_value

    def act(self, **ops) -> Any:
        """
        Perform an action
        :return:
        """
        # get the values of the column
        col_vals = ops['data']

        # generalize the data given
        for i, item in enumerate(ops["data"]):
            col_vals[i] = self.transform

        ops["data"] = col_vals
        return ops['data']


class ActionSuppress(ActionBase, WithHierarchyTable):
    """
    Implements the suppress action
    """
    def __init__(self, column_name: str, suppress_table: Hierarchy):
        super(ActionSuppress, self).__init__(column_name=column_name, action_type=ActionType.SUPPRESS)

        self.table = suppress_table

    def act(self, **ops) -> None:
        """
        Perform the action
        :return: None
        """

        # get the values of the column
        col_vals = ops['data'] #.values

        # generalize the data given
        for i, item in enumerate(ops["data"]):

            value = self.table[item] #.value
            col_vals[i] = value

        ops["data"] = col_vals
        return ops['data']


class ActionStringGeneralize(ActionBase, WithHierarchyTable):
    """Implements the generalization action. The generalization_table
    must implement the __getitem__ function

    """

    def __init__(self, column_name: str, generalization_table:  Hierarchy) -> None:
        """Constructor

        Parameters
        ----------
        column_name : The column name this action is acting on
        generalization_table : The hierarchy for the generalization

        """
        super(ActionStringGeneralize, self).__init__(column_name=column_name, action_type=ActionType.GENERALIZE)
        self.table = generalization_table

    def act(self, **ops) -> Any:
        """Performs the action

        Parameters
        ----------
        ops : The data to distort

        Returns
        -------

        The distorted data

        """

        # get the values of the column
        col_vals = ops['data'] #.values

        # generalize the data given
        for i, item in enumerate(col_vals):

            # How do we update the generalizations?
            value = self.table[item]
            col_vals[i] = value

        ops["data"] = col_vals

        return ops['data']

    def add(self, key: Any, value: Any) -> None:
        """Add a new item in the underlying hierarchy

        Parameters
        ----------
        key : The key to use for the new item
        value : The value of the new item

        Returns
        -------

        None
        """
        self.table.add(key, value)


class ActionNumericBinGeneralize(ActionBase, WithHierarchyTable):
    """Generalization Action for numeric columns using bins
    """

    def __init__(self, column_name: str, generalization_table:  Hierarchy):
        """
        Constructor
        Parameters
        ----------
        column_name : The name of the column this is acting on
        generalization_table : The bins to use
        """
        super(ActionNumericBinGeneralize, self).__init__(column_name=column_name, action_type=ActionType.GENERALIZE)

        self.table = generalization_table
        self.bins = []

        start = self.table[0]
        for i in range(1, len(self.table)):
            self.bins.append((start, self.table[i]))
            start = self.table[i]

    def act(self, **ops) -> Any:
        """
        Perform the action
        Parameters
        ----------
        ops : The data to distort

        Returns
        -------
        Typically the action returns the distorted subset of the data
        """

        # get the values of the column
        col_vals = ops['data'] #.values

        # generalize the data given
        for i, item in enumerate(col_vals):

            # find out the bin it belongs to
            bin_idx = np.digitize(item, self.table)

            if bin_idx == 0 or bin_idx == len(self.table):
                # this means data is out of bounds
                raise ValueError("Invalid bin index for value {0}. "
                                 "Bin index={1} not in [1, {2}]".format(item, bin_idx, len(self.table)))

            low = self.bins[bin_idx - 1][0]
            high = self.bins[bin_idx - 1][1]

            # How do we update the generalizations?
            # use
            value = (high + low)*0.5
            col_vals[i] = value

        ops["data"] = col_vals
        return ops['data']


class ActionNumericStepGeneralize(ActionBase):

    def __init__(self, column_name: str, step: float):
        super(ActionNumericStepGeneralize, self).__init__(column_name=column_name, action_type=ActionType.GENERALIZE)
        self.step = step

    def act(self, **ops):
        """
        Perform an action
        :return:
        """

        # get the values of the column
        col_vals = ops['data']

        # generalize the data given
        for i, item in enumerate(col_vals):
            value = item + self.step*item
            col_vals[i] = value

        ops["data"] = col_vals
        return ops['data']









