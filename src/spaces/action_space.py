"""
ActionSpace class. This is a wrapper to the discrete
actions in the actions.py module
"""

from gym.spaces.discrete import Discrete
from src.spaces.actions import ActionBase


class ActionSpace(Discrete):
    """
    ActionSpace class models a discrete action space of size n
    """

    def __init__(self, n: int) -> None:

        super(ActionSpace, self).__init__(n=n)

        # the list of actions the space contains
        self.actions = []

    def __getitem__(self, item) -> ActionBase:
        """
        Returns the item-th action
        :param item: The index of the action to return
        :return: An action obeject
        """
        return self.actions[item]

    def __setitem__(self, key: int, value: ActionBase) -> None:
        """
        Update the key-th Action with the new value
        :param key: The index to the action to update
        :param value: The new action
        :return: None
        """
        self.actions[key] = value

    def add(self, action: ActionBase) -> None:
        """
        Add a new action in the space
        :param action:
        :return:
        """

        if len(self.actions) >= self.n:
            raise ValueError("Action space is saturated. You cannot add a new action")

        # set a valid id for the action
        action.idx = len(self.actions)
        self.actions.append(action)

    def add_many(self, *actions) -> None:
        """
        Add many actions in one go
        :param actions: List of actions to add
        :return: None
        """
        for a in actions:
            self.add(action=a)

    def sample_and_get(self) -> ActionBase:
        """
        Sample the space and return an action to the application
        :return: The sampled action
        """
        action_idx = self.sample()
        return self.actions[action_idx]
