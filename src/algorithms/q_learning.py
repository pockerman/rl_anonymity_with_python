"""
Simple Q-learning algorithm
"""

import numpy as np
from typing import TypeVar

from src.exceptions.exceptions import InvalidParamValue
from src.utils.mixins import WithMaxActionMixin, WithQTableMixinBase

Env = TypeVar('Env')
Policy = TypeVar('Policy')
Criterion = TypeVar('Criterion')


class QLearnConfig(object):
    """
    Configuration  for Q-learning
    """
    def __init__(self):
        self.gamma: float = 1.0
        self.alpha: float = 0.1
        self.n_itrs_per_episode: int = 100
        self.policy: Policy = None


class QLearning(WithMaxActionMixin):
    """
    Q-learning algorithm implementation
    """

    def __init__(self, algo_config: QLearnConfig):
        super(QLearning, self).__init__()
        self.q_table = {}
        self.config = algo_config

    @property
    def name(self) -> str:
        return "QLearn"

    def actions_before_training(self, env: Env, **options):

        if not isinstance(self.config.policy, WithQTableMixinBase):
            raise InvalidParamValue(param_name="policy", param_value=str(self.config.policy))

        for state in range(1, env.n_states):
            for action in range(env.n_actions):
                self.q_table[state, action] = 0.0

    def actions_after_episode_ends(self, **options):
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """

        self.config.policy.actions_after_episode(options['episode_idx'])

    def play(self, env: Env, stop_criterion: Criterion) -> None:
        """
        Play the game on the environment. This should produce
        a distorted dataset
        :param stop_criterion:
        :param env:
        :return:
        """

        # loop over the columns and for the
        # column get the action that corresponds to
        # the max payout.
        # TODO: This will no work as the distortion is calculated
        # by summing over the columns.

        # set the q_table for the policy
        self.config.policy.q_table = self.q_table
        total_dist = env.total_current_distortion()
        while stop_criterion.continue_itr(total_dist):

            if stop_criterion.iteration_counter == 12:
                print("Break...")

            # use the policy to select an action
            state_idx = env.get_aggregated_state(total_dist)
            action_idx = self.config.policy.on_state(state_idx)
            action = env.get_action(action_idx)
            print("{0} At state={1} with distortion={2} select action={3}".format("INFO: ", state_idx, total_dist,
                                                                                  action.column_name + "-" + action.action_type.name))
            env.step(action=action)
            total_dist = env.total_current_distortion()

    def on_episode(self, env: Env, **options) -> tuple:

        # episode score
        episode_score = 0
        counter = 0
        total_distortion = 0

        time_step = env.reset()
        state = time_step.observation

        for itr in range(self.config.n_itrs_per_episode):

            # epsilon-greedy action selection
            action_idx = self.config.policy(q_table=self.q_table, state=state)

            action = env.get_action(action_idx)

            # take action A, observe R, S'
            next_time_step = env.step(action)
            next_state = next_time_step.observation
            reward = next_time_step.reward

            # add reward to agent's score
            episode_score += reward
            self._update_Q_table(state=state, action=action_idx, reward=reward,
                                 next_state=next_state, n_actions=env.n_actions)
            state = next_state  # S <- S'
            counter += 1
            total_distortion += next_time_step.info["total_distortion"]

            if next_time_step.last():
                break

        return episode_score, total_distortion, counter

    def _update_Q_table(self, state: int, action: int, n_actions: int,
                        reward: float, next_state: int = None) -> None:
        """
        Update the Q-value for the state
        """

        # estimate in Q-table (for current state, action pair)
        q_s = self.q_table[state, action]

        # value of next state
        Qsa_next = \
            self.q_table[next_state, self.max_action(next_state, n_actions=n_actions)] if next_state is not None else 0
        # construct TD target
        target = reward + (self.config.gamma * Qsa_next)

        # get updated value
        new_value = q_s + (self.config.alpha * (target - q_s))
        self.q_table[state, action] = new_value
