"""module reward_manager specifies a class that handles
the rewards awarded by the environment.

"""

from typing import TypeVar, Any


State = TypeVar("State")
Action = TypeVar("Action")


class RewardManager(object):
    """The RewardManager class

    """
    def __init__(self, bounds: tuple, out_of_max_bound_reward: float,
                 out_of_min_bound_reward: float,
                 in_bounds_reward: float, punish_factor: float,
                 min_distortions: Any, max_distortions: Any) -> None:
        self.bounds = bounds
        self.out_of_max_bound_reward: float = out_of_max_bound_reward
        self.out_of_min_bound_reward = out_of_min_bound_reward
        self.in_bounds_reward = in_bounds_reward
        self.min_distortions = min_distortions
        self.max_distortions = max_distortions
        self.punish_factor = punish_factor

    def get_reward_for_state(self, total_distortion: float,
                             current_state: State,
                             next_state: State,
                             min_dist_bins: Any,
                             **options) -> float:
        """Returns a user specified reward signal depending on the
        state and the options given

        Parameters
        ----------
        state
        options

        Returns
        -------

        """

        if total_distortion < self.bounds[0]:
            return self.punish_factor * self.out_of_min_bound_reward
        if total_distortion > self.bounds[1]:
            return self.punish_factor * self.out_of_max_bound_reward

        """
        # TODO: these modifications will cause the agent to always
        # move close to transition points Also need to account for
        # tuple states
        if next_state is not None and current_state is not None:
            if next_state < min_dist_bin <= self.current_time_step.observation:
                # the agent chose to step into the chaos again
                # we punish him with double the reward
                reward = self.config.punish_factor * self.config.reward_manager.out_of_min_bound_reward
            elif next_state > max_dist_bin >= self.current_time_step.observation:
                # the agent is going to chaos from above
                # punish him
                reward = self.config.punish_factor * self.config.reward_manager.out_of_max_bound_reward

            elif next_state >= min_dist_bin > self.current_time_step.observation:
                # the agent goes towards the transition of min point so give a higher reward
                # for this
                reward = self.config.reward_factor * self.config.reward_manager.in_bounds_reward

            elif next_state <= max_dist_bin < self.current_time_step.observation:
                # the agent goes towards the transition of max point so give a higher reward
                # for this
                reward = self.config.reward_factor * self.config.reward_manager.in_bounds_reward

        if total_distortion > self.bounds[1]:
            return self.out_of_max_bound_reward

        if total_distortion < self.bounds[0]:
            return self.out_of_min_bound_reward
        """

        return self.in_bounds_reward

