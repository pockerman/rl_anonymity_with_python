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

    def transform(self) -> bool:
        return self is ActionType.TRANSFORM

    def suppress(self) -> bool:
        return self is ActionType.SUPPRESS

    def generalize(self) -> bool:
        return self is ActionType.GENERALIZE


class ActionBase(metaclass=abc.ABCMeta):

    def __init__(self, action_type: ActionType) -> None:
        self.action_type = action_type

    @abc.abstractmethod
    def act(self, **ops):
        """
        Perform an action
        :return:
        """


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


class ActionSuppress(ActionBase):

    """
    Implements the suppress action
    """
    def __init__(self):
        super(ActionSuppress, self).__init__(action_type=ActionType.SUPPRESS)

    def act(self, **ops):
        """
        Perform an action
        :return:
        """
        pass


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


