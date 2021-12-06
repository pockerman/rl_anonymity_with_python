import abc
import enum
from typing import List

from utils.hierarchy_base import HierarchyBase


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

    def __init__(self, action_type: ActionType) -> None:
        self.action_type = action_type

    @abc.abstractmethod
    def act(self, **ops) -> None:
        """
        Perform an action
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

    def __init__(self) -> None:
        super(ActionIdentity, self).__init__(action_type=ActionType.IDENTITY)

    def act(self, **ops):
        """
        Perform the action
        :return:
        """
        pass


class ActionTransform(ActionBase):

    """
    Implements the transform action
    """
    def __init__(self):
        super(ActionTransform, self).__init__(action_type=ActionType.TRANSFORM)

    def act(self, **ops):
        """
        Perform an action
        :return:
        """
        pass


class ActionSuppress(ActionBase, _WithTable):

    """
    Implements the suppress action
    """
    def __init__(self, suppress_table=None):
        super(ActionSuppress, self).__init__(action_type=ActionType.SUPPRESS)

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


class ActionGeneralize(ActionBase):
    """
    Implements the generalization action
    """

    def __init__(self):
        super(ActionGeneralize, self).__init__(action_type=ActionType.GENERALIZE)
        self.generalization_table = {}

    def act(self, **ops):
        """
        Perform an action
        :return:
        """
        # generalize the data given
        for item in ops["data"]:

            # How do we update the generalizations?
            value = self.generalization_table[item].value
            item = value

        # update the generalization
        self._move_next()

    def add_generalization(self, key: str, values: HierarchyBase) -> None:
        self.generalization_table[key] = values

    def _move_next(self):

        for item in self.generalization_table:
            next(self.generalization_table[item])


