"""
Implementation of SARSA semi-gradient algorithm.
Initial implementation is inspired from
https://michaeloneill.github.io/RL-tutorial.html
"""
import numpy as np
from typing import TypeVar
from dataclasses import dataclass

from src.utils.mixins import WithMaxActionMixin, WithQTableMixinBase
from src.utils.episode_info import EpisodeInfo

from src.algorithms.q_estimator import QEstimator
from src.exceptions.exceptions import InvalidParamValue

Env = TypeVar('Env')
Criterion = TypeVar('Criterion')
Policy = TypeVar('Policy')
Estimator = TypeVar('Estimator')

@dataclass(init=True, repr=True)
class SARSAnConfig:
    """Configuration class for n-step SARSA algorithm
    """
    gamma: float = 1.0
    alpha: float = 0.1
    n: int = 10
    n_itrs_per_episode: int = 100
    max_size: int = 4096
    use_trace: bool = False
    policy: Policy = None
    estimator: Estimator = None
    reset_estimator_z_traces: bool = False


class SARSAn(WithMaxActionMixin):
    """Implementation of n-step  semi-gradient SARSA algorithm
    """

    def __init__(self, sarsa_config: SARSAnConfig):
        super(SARSAn, self).__init__(table={})
        self.name = "SARSAn"
        self.config = sarsa_config

    def play(self, env: Env, stop_criterion: Criterion) -> None:
        """
        Apply the trained agent on the given environment.

        Parameters
        ----------
        env: The environment to apply the agent
        stop_criterion: Criteria that specify when play should stop

        Returns
        -------

        None

        """
        # loop over the columns and for the
        # column get the action that corresponds to
        # the max payout.
        # TODO: This will no work as the distortion is calculated
        # by summing over the columns.

        # set the q_table for the policy
        # this is the table we should be using to
        # make decisions
        self.config.policy.q_table = self.q_table
        total_dist = env.total_current_distortion()
        while stop_criterion.continue_itr(total_dist):
            # use the policy to select an action
            state_idx = env.get_aggregated_state(total_dist)
            action_idx = self.config.policy.on_state(state_idx)
            action = env.get_action(action_idx)
            print("{0} At state={1} with distortion={2} select action={3}".format("INFO: ", state_idx, total_dist,
                                                                                  action.column_name + "-" + action.action_type.name))
            env.step(action=action)
            total_dist = env.total_current_distortion()

    def actions_before_training(self, env: Env) -> None:
        """
        Any action to execute before
        entering the training loop
        :param env:
        :return:
        """
        self._validate(env)

        # reset the estimator
        self.config.estimator.reset(self.config.reset_estimator_z_traces)

    def actions_before_episode_begins(self, **options) -> None:
        """
        Actions for the agent to perform
        :param options:
        :return:
        """
        # reset the estimator
        self.config.estimator.reset(self.config.reset_estimator_z_traces)

    def actions_after_episode_ends(self, **options) -> None:
        pass

    def on_episode(self, env: Env) -> EpisodeInfo:
        """
        Train the agent on the given algorithm
        :param env:
        :return:
        """

        # reset before the episode begins
        time_step = env.reset()
        state = time_step.observation

        action_idx = self.config.policy(self.q_table, state)
        action = env.get_action(action_idx)

        # vars to measure performance
        episode_score = 0
        counter = 0
        total_distortion = 0
        T = float('inf')
        actions = [action_idx]
        rewards = [0.0]
        states = [state]
        for itr in range(self.config.n_itrs_per_episode):

            if itr < T:

                # take action A, observe R, S'
                next_time_step = env.step(action)
                next_state = next_time_step.observation
                states.append(next_state)
                reward = next_time_step.reward

                total_distortion += next_time_step.info["total_distortion"]
                episode_score += reward
                rewards.append(reward)

                if next_time_step.done:
                    T = itr + 1
                else:

                    next_action_idx = self.config.policy(self.q_table, next_state)
                    next_action = env.get_action(next_action_idx)
                    actions.append(next_action_idx)

            # should we update
            update_time = itr + 1 - self.config.n
            if update_time >= 0:

                # build target
                target = 0
                for i in range(update_time + 1, min(T, update_time + self.config.n) + 1):
                    target += np.power(self.config.gamma, i - update_time - 1) * rewards[i]

                if update_time + self.config.n < T:
                    q_values_next = self.config.estimator.predict(states[update_time + self.config.n])
                    target += q_values_next[actions[update_time + self.config.n]]

                # Update step. what happens if the update_time is greater than
                # len(states) or len(actions)

                if update_time >= len(states) or update_time >= len(actions):
                    raise InvalidParamValue(param_name="update_time", param_value=str(update_time))

                # update the state for the respective action
                # with the computed target
                self.config.estimator.update(states[update_time], actions[update_time], target)

            if update_time == T - 1:
                break

            counter += 1
            state = next_state
            action = next_action

        episode_info = EpisodeInfo()
        episode_info.episode_score = episode_score
        episode_info.total_distortion = total_distortion
        episode_info.info["n_iterations"] = counter
        return episode_info

    def _validate(self, env: Env) -> None:
        """
        Performs necessary checks
        :param env:
        :return:
        """
        # validate
        is_tiled = getattr(env, "IS_TILED_ENV_CONSTRAINT", None)
        if is_tiled is None or is_tiled is False:
            raise ValueError("The given environment does not "
                             "satisfy the IS_TILED_ENV_CONSTRAINT constraint")

        if not isinstance(self.config.policy, WithQTableMixinBase):
            raise InvalidParamValue(param_name="policy", param_value=str(self.config.policy))

        if self.config.estimator is None:
            raise ValueError("Estimator has not been set")

