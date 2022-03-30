"""Module action_space
Specifies  a wrapper to the discrete
actions in the actions.py module

"""

import numpy as np
import random
from gym.spaces.discrete import Discrete
from src.spaces.actions import ActionBase, ActionType


class ActionSpace(Discrete):
    """ActionSpace class models a discrete action space of size n
    """

    @classmethod
    def from_actions(cls, *actions: ActionBase):

        space = cls(n=len(actions))
        space.add_many(*actions)
        return space

    def __init__(self, n: int) -> None:
        """Constructor

        Parameters
        ----------
        n: The size of the space

        """

        super(ActionSpace, self).__init__(n=n)

        # the list of actions the space contains
        self.actions = []

    def __getitem__(self, idx: int) -> ActionBase:
        """Returns the idx-th action

        Parameters
        ----------
        idx: The action index to return

        Returns
        -------

        An instance of ActionBase
        """

        return self.actions[idx]

    def __setitem__(self, idx: int, action: ActionBase) -> None:
        """Set the idx-th action

        Parameters
        ----------
        idx: The index of the action to set
        action: The action value to set

        Returns
        -------

        None
        """

        self.actions[idx] = action

    def __len__(self) -> int:
        return len(self.actions)

    def shuffle(self, seed: int = 42) -> None:
        """Shuffles the action list

        Returns
        -------
        None
        """

        random.seed(seed)
        random.shuffle(self.actions)

        # fix the ids of the actions to
        # correspond to the shuffling
        for i in range(len(self.actions)):
            self.actions[i].idx = i

    def get_action_by_name_and_type(self, column_name: str, action_type: ActionType) -> ActionBase:
        """Returns the action that has the given type and the
        given column name

        Parameters
        ----------

        column_name: The column name
        action_type: The action type

        Returns
        -------

        An instance of ActionBase
        """

        for action in self.actions:
            if action.column_name == column_name and \
                    action.action_type == action_type:
                return action

        raise ValueError("No action exists for column={0} with type {1}".format(column_name, action_type.name))

    def add(self, action: ActionBase) -> None:
        """Add a new action

        Parameters
        ----------
        action: The action to add

        Returns
        -------

        None
        """

        if len(self.actions) >= self.n:
            raise ValueError("Action space is saturated. You cannot add a new action")

        # set a valid id for the action
        action.idx = len(self.actions)
        self.actions.append(action)

    def add_many(self, *actions) -> None:
        """Add many actions

        Parameters
        ----------
        actions: The list of action to add

        Returns
        -------

        None
        """

        for a in actions:
            self.add(action=a)

    def sample_and_get(self) -> ActionBase:
        """Sample an index and return the relevant action

        Returns
        -------

        An instance to ActionBase

        """

        action_idx = self.sample()
        return self.actions[action_idx]


