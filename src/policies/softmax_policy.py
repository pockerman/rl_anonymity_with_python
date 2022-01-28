import numpy as np
from typing import TypeVar, Any
from src.utils.mixins import WithQTableMixin

QTable = TypeVar('QTable')


class SoftMaxPolicy(WithQTableMixin):

    def __init__(self, n_actions: int, tau: float) -> None:
        self.n_actions = n_actions
        self.tau = tau

    def __str__(self) -> str:
        return "SoftMaxPolicy"

    def __call__(self, q_table: QTable, state: Any) -> int:
        """
        Execute the policy
        :param q_table:
        :param state:
        :return:
        """
        self.q_table = q_table
        action_values = [q_table[state, a] for a in range(self.n_actions)]
        softmax = np.exp(np.array(action_values) / self.tau) / np.sum( np.exp(np.array(action_values) / self.tau) )

        # return the action index by choosing from
        return np.random.choice( [a for a in range(self.n_actions)], p=softmax)

    def on_state(self, state: Any) -> int:
        """
        Returns the optimal action on the current state
        :param state:
        :return:
        """
        action_values = [self.q_table[state, a] for a in range(self.n_actions)]
        softmax = np.exp(np.array(action_values) / self.tau) / np.sum(np.exp(np.array(action_values) / self.tau))

        # return the action index by choosing from
        return np.random.choice([a for a in range(self.n_actions)], p=softmax)

    def actions_after_episode(self, episode_idx: int, **options) -> None:
        pass