"""
Epsilon greedy policy implementation
"""
import random
import numpy as np
from enum import Enum
from typing import Any, TypeVar
from dataclasses import dataclass

from src.utils.mixins import WithMaxActionMixin

UserDefinedDecreaseMethod = TypeVar('UserDefinedDecreaseMethod')
Env = TypeVar("Env")
QTable = TypeVar("QTable")


class EpsilonDecayOption(Enum):
    """
    Options for reducing epsilon
    """

    NONE = 0
    EXPONENTIAL = 1
    INVERSE_STEP = 2
    CONSTANT_RATE = 3
    USER_DEFINED = 4


@dataclass(init=True, repr=True)
class EpsilonGreedyConfig(object):
    """Configuration class for EpsilonGreedyPolicy

    """
    eps: float = 1.0
    n_actions: int = 1
    decay_op: EpsilonDecayOption = EpsilonDecayOption.NONE
    max_eps: float = 1.0
    min_eps: float = 0.001
    epsilon_decay_factor: float = 0.01
    user_defined_decrease_method: UserDefinedDecreaseMethod = None


class EpsilonGreedyPolicy(WithMaxActionMixin):
    """Epsilon-greedy policy implementation
    """

    @classmethod
    def from_config(cls, config: EpsilonGreedyConfig):
        return cls(eps=config.eps, n_actions=config.n_actions,
                   decay_op=config.decay_op, min_eps=config.min_eps,
                   max_eps=config.max_eps, epsilon_decay_factor=config.epsilon_decay_factor,
                   user_defined_decrease_method=config.user_defined_decrease_method)

    def __init__(self, eps: float, n_actions: int,
                 decay_op: EpsilonDecayOption,
                 max_eps: float = 1.0, min_eps: float = 0.001,
                 epsilon_decay_factor: float = 0.01,
                 user_defined_decrease_method: UserDefinedDecreaseMethod = None) -> None:
        super(WithMaxActionMixin, self).__init__(table={})
        self._eps = eps
        self._n_actions = n_actions
        self._decay_op = decay_op
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._epsilon_decay_factor = epsilon_decay_factor
        self.user_defined_decrease_method: UserDefinedDecreaseMethod = user_defined_decrease_method

    def __str__(self) -> str:
        return self.__name__

    def __call__(self, q_table: QTable, state: Any) -> int:
        """
        Execute the policy
        :param q_func:
        :param state:
        :return:
        """

        # update the store q_table
        self.q_table = q_table

        # select greedy action with probability epsilon
        if random.random() > self._eps:
            return self.max_action(state=state, n_actions=self._n_actions)
        else:

            # otherwise, select an action randomly
            # what happens if we select an action that
            # has exhausted it's transforms?
            return random.choice(np.arange(self._n_actions))

    def on_state(self, state: Any) -> int:
        """
        Returns the optimal action on the current state
        :param state:
        :return:
        """
        return self.max_action(state=state, n_actions=self._n_actions)

    def actions_after_episode(self, episode_idx: int, **options) -> None:
        """
        Apply actions on the policy after the end of the episode
        :param episode_idx: The episode index
        :param options:
        :return: None
        """

        if self._decay_op == EpsilonDecayOption.NONE:
            return

        if self._decay_op == EpsilonDecayOption.USER_DEFINED:
            self._eps = self.user_defined_decrease_method(self._eps, episode_idx)

        if self._decay_op == EpsilonDecayOption.INVERSE_STEP:

            if episode_idx == 0:
                episode_idx = 1

            self._eps = 1.0 / episode_idx

        elif self._decay_op == EpsilonDecayOption.EXPONENTIAL:
            self._eps = self._min_eps + (self._max_eps - self._min_eps) * np.exp(-self._epsilon_decay_factor * episode_idx)

        elif self._decay_op == EpsilonDecayOption.CONSTANT_RATE:
            self._eps -= self._epsilon_decay_factor

        if self._eps < self._min_eps:
            self._eps = self._min_eps


