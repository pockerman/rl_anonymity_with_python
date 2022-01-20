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
    def __init__(self, bounds: tuple, out_of_max_bound_reward: float,
                 out_of_min_bound_reward: float,
                 in_bounds_reward: float) -> None:
        self.bounds = bounds
        self.out_of_max_bound_reward: float = out_of_max_bound_reward
        self.out_of_min_bound_reward = out_of_min_bound_reward
        self.in_bounds_reward = in_bounds_reward

    def get_reward_for_state(self, state: float, **options) -> float:

        if state > self.bounds[1]:
            return self.out_of_max_bound_reward

        if state < self.bounds[0]:
            return self.out_of_min_bound_reward

        return self.in_bounds_reward

