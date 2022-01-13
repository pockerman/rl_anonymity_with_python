"""
Various utilities to handle reward assignment
"""

from typing import TypeVar


State = TypeVar("State")
Action = TypeVar("Action")


class RewardManager(object):
    """
    Helper class to assign rewards
    """
    def __init__(self, average_distortion_constraint: dict) -> None:
        self.average_distortion_constraint: dict = average_distortion_constraint

    def get_state_reward(self, state_name: str, action: Action, state_distortion: float) -> float:
        """
        Returns the reward associated with the action
        applied
        :param options:
        :return:
        """

        if state_name not in self.average_distortion_constraint:
            raise KeyError("state {0} does not exist".format(state_name))

        state_rewards = self.average_distortion_constraint[state_name]

        if state_distortion < state_rewards[0]:
            return state_rewards[1]

        return state_rewards[2]