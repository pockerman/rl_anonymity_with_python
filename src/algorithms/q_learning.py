"""
Simple Q-learning algorithm
"""

import numpy as np
from typing import TypeVar

from src.exceptions.exceptions import InvalidParamValue
from src.utils.mixins import WithMaxActionMixin

Env = TypeVar('Env')
Policy = TypeVar('Policy')


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

        if self.config.policy is None:
            raise InvalidParamValue(param_name="policy", param_value="None")

        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                self.q_table[state, action] = 0.0

    def actions_after_episode_ends(self, **options):
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """

        self.config.policy.actions_after_episode(options['episode_idx'])

    def train(self, env: Env, **options) -> tuple:

        # episode score
        episode_score = 0  # initialize score
        counter = 0

        time_step = env.reset()
        state = time_step.observation

        for itr in range(self.config.n_itrs_per_episode):

            # epsilon-greedy action selection
            action_idx = self.config.policy(q_func=self.q_table, state=state)

            action = env.get_action(action_idx)

            # take action A, observe R, S'
            next_time_step = env.step(action)
            next_state = next_time_step.observation
            reward = next_time_step.reward

            next_state_id = next_state.state_id if next_state is not None else None

            # add reward to agent's score
            episode_score += next_time_step.reward
            self._update_Q_table(state=state.state_id, action=action_idx, reward=reward,
                                 next_state=next_state_id, n_actions=env.action_space.n)
            state = next_state  # S <- S'
            counter += 1

            if next_time_step.last():
                break

        return episode_score, counter

    def _update_Q_table(self, state: int, action: int, n_actions: int, reward: float, next_state: int = None) -> None:
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
