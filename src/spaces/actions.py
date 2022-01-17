import abc
import enum
from typing import List

from src.utils.hierarchy_base import HierarchyBase
from src.utils.mixins import WithHierarchyTable


def move_next(iterators: List) -> None:
    """
    Loop over the iterators and move them
    to the next item
    :param iterators: The list of iterators to propagate
    :return: None
    """
    for item in iterators:
        next(item)


class ActionType(enum.IntEnum):
    """
    Defines the status of an Action
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
    def get_maximum_number_of_transforms(self) -> int:
        """
        Returns the maximum number of transforms that the action applies
        :return:
        """

    @abc.abstractmethod
    def is_exhausted(self) -> bool:
        """
        Returns true if the action has exhausted all its
        transforms
        :return:
        """

    @abc.abstractmethod
    def reinitialize(self) -> None:
        """
        Reinitialize the action to the state when the
        constructor is called
        :return:
        """


class ActionIdentity(ActionBase):
    """
    Implements the identity action
    """

    def __init__(self, column_name: str) -> None:
        super(ActionIdentity, self).__init__(column_name=column_name, action_type=ActionType.IDENTITY)
        self.called = False

    def act(self, **ops) -> None:
        """
        Perform the action
        :return:
        """
        self.called = True

    def get_maximum_number_of_transforms(self):
        """
        Returns the maximum number of transforms that the action applies
        :return:
        """
        return 1

    def is_exhausted(self) -> bool:
        """
        Returns true if the action has exhausted all its
        transforms
        :return:
        """
        return self.called

    def reinitialize(self) -> None:
        """
        Reinitialize the action to the state when the
        constructor is called
        :return:
        """
        self.called = False


class ActionRestore(ActionBase, WithHierarchyTable):
    """
    Implements the restore action
    """

    def __init__(self, column_name: str, restore_table):
        super(ActionRestore, self).__init__(column_name=column_name, action_type=ActionType.RESTORE)
        self.table = restore_table

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

    def is_exhausted(self) -> bool:
        """
        Returns true if the action has exhausted all its
        transforms
        :return:
        """
        raise NotImplementedError("Method not implemented")

    def reinitialize(self) -> None:
        """
        Reinitialize the action to the state when the
        constructor is called
        :return:
        """
        raise NotImplementedError("Method not implemented")


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

    def is_exhausted(self) -> bool:
        """
        Returns true if the action has exhausted all its
        transforms
        :return:
        """
        raise NotImplementedError("Method not implemented")

    def reinitialize(self) -> None:
        """
        Reinitialize the action to the state when the
        constructor is called
        :return:
        """
        raise NotImplementedError("Method not implemented")


class ActionSuppress(ActionBase, WithHierarchyTable):
    """
    Implements the suppress action
    """
    def __init__(self, column_name: str, suppress_table=None):
        super(ActionSuppress, self).__init__(column_name=column_name, action_type=ActionType.SUPPRESS)

        self.table = suppress_table

        # fill in the iterators
        self.iterators = [iter(self.table[item]) for item in self.table]

    def act(self, **ops) -> None:
        """
        Perform the action
        :return: None
        """

        # get the values of the column
        col_vals = ops['data'].values

        # generalize the data given
        for i, item in enumerate(ops["data"]):

            value = self.table[item].value
            col_vals[i] = value

        ops["data"] = col_vals

        # update the generalization iterators
        # so next time we visit we update according to
        # the new values
        move_next(iterators=self.iterators)
        return ops['data']

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

    def is_exhausted(self) -> bool:
        """
        Returns true if the action has exhausted all its
        transforms
        :return:
        """
        return self.finished()

    def reinitialize(self) -> None:
        """
        Reinitialize the action to the state when the
        constructor is called
        :return:
        """
        self.reset_iterators()


class ActionGeneralize(ActionBase, WithHierarchyTable):
    """
    Implements the generalization action
    """

    def __init__(self, column_name: str, generalization_table: dict = None):
        super(ActionGeneralize, self).__init__(column_name=column_name, action_type=ActionType.GENERALIZE)

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

    def is_exhausted(self) -> bool:
        """
        Returns true if the action has exhausted all its
        transforms
        :return:
        """
        return self.finished()

    def reinitialize(self) -> None:
        """
        Reinitialize the action to the state when the
        constructor is called
        :return:
        """
        self.reset_iterators()




