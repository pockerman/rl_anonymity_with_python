import abc
import enum
from typing import List

from src.utils.hierarchy_base import HierarchyBase


class ActionType(enum.IntEnum):
    """
    Defines the status of an Action
    """

    TRANSFORM = 0
    SUPPRESS = 1
    GENERALIZE = 2
    IDENTITY = 3

    def transform(self) -> bool:
        return self is ActionType.TRANSFORM

    def suppress(self) -> bool:
        return self is ActionType.SUPPRESS

    def generalize(self) -> bool:
        return self is ActionType.GENERALIZE

    def identity(self) -> bool:
        return self is ActionType.IDENTITY


class ActionBase(metaclass=abc.ABCMeta):
    """
    Base class for actions
    """

    def __init__(self, column_name: str, action_type: ActionType) -> None:
        self.column_name = column_name
        self.action_type = action_type
        self.idx = None
        self.key = (self.column_name, self.action_type)

    @abc.abstractmethod
    def act(self, **ops) -> None:
        """
        Perform an action
        :return:
        """

    @abc.abstractmethod
    def get_maximum_number_of_transforms(self):
        """
        Returns the maximum number of transforms that the action applies
        :return:
        """


def move_next(iterators: List) -> None:
    """
    Loop over the iterators and move them
    to the next item
    :param iterators: The list of iterators to propagate
    :return: None
    """
    for item in iterators:
        next(item)


class _WithTable(object):

    def __init__(self) -> None:
        super(_WithTable, self).__init__()
        self.table = {}
        self.iterators = []

    def add_hierarchy(self, key: str, hierarchy: HierarchyBase) -> None:
        """
        Add a hierarchy for the given key
        :param key: The key to attach the Hierarchy
        :param hierarchy: The hierarchy to attach
        :return: None
        """
        self.table[key] = hierarchy


class ActionIdentity(ActionBase):
    """
    Implements the identity action
    """

    def __init__(self, column_name: str) -> None:
        super(ActionIdentity, self).__init__(column_name=column_name, action_type=ActionType.IDENTITY)

    def act(self, **ops):
        """
        Perform the action
        :return:
        """
        pass

    def get_maximum_number_of_transforms(self):
        """
        Returns the maximum number of transforms that the action applies
        :return:
        """
        return 1


class ActionTransform(ActionBase):

    """
    Implements the transform action
    """
    def __init__(self, column_name: str):
        super(ActionTransform, self).__init__(column_name=column_name, action_type=ActionType.TRANSFORM)

    def act(self, **ops):
        """
        Perform an action
        :return:
        """
        pass

    def get_maximum_number_of_transforms(self):
        """
        Returns the maximum number of transforms that the action applies
        :return:
        """
        raise NotImplementedError("Method not implemented")


class ActionSuppress(ActionBase, _WithTable):

    """
    Implements the suppress action
    """
    def __init__(self, column_name: str, suppress_table=None):
        super(ActionSuppress, self).__init__(column_name=column_name, action_type=ActionType.SUPPRESS)

        if suppress_table is not None:
            self.table = suppress_table

        # fill in the iterators
        self.iterators = [iter(self.table[item]) for item in self.table]

    def act(self, **ops) -> None:
        """
        Perform the action
        :return: None
        """

        # generalize the data given
        for i, item in enumerate(ops["data"]):

            if item in self.table:
                value = self.table[item].value
                item = value
                ops["data"][i] = value

        # update the generalization
        move_next(iterators=self.iterators)

    def get_maximum_number_of_transforms(self):
        """
        Returns the maximum number of transforms that the action applies
        :return:
        """
        max_transform = 0

        for item in self.table:
            size = len(self.table[item])

            if size > max_transform:
                max_transform = size

        return max_transform


class ActionGeneralize(ActionBase, _WithTable):
    """
    Implements the generalization action
    """

    def __init__(self, column_name: str, generalization_table: dict = None):
        super(ActionGeneralize, self).__init__(column_name=column_name, action_type=ActionType.GENERALIZE)

        if generalization_table is not None:
            self.table = generalization_table

        # fill in the iterators
        self.iterators = [iter(self.table[item]) for item in self.table]

    def act(self, **ops):
        """
        Perform an action
        :return:
        """

        # get the values of the column
        col_vals = ops['data'].values

        # generalize the data given
        for i, item in enumerate(col_vals):

            #print(item)
            # How do we update the generalizations?
            value = self.table[item].value
            col_vals[i] = value

        ops["data"] = col_vals

        # update the generalization iterators
        # so next time we visit we update according to
        # the new values
        move_next(iterators=self.iterators)
        return ops['data']

    def add_generalization(self, key: str, values: HierarchyBase) -> None:
        self.table[key] = values

    def get_maximum_number_of_transforms(self):
        """
        Returns the maximum number of transforms that the action applies
        :return:
        """
        max_transform = 0

        for item in self.table:
            size = len(self.table[item])

            if size > max_transform:
                max_transform = size

        return max_transform




