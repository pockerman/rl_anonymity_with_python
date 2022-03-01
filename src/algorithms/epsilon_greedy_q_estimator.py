"""Module epsilon_greedy_q_estimator

"""
from typing import TypeVar
import numpy as np
from dataclasses import dataclass

from src.utils.mixins import WithEstimatorMixin
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonGreedyConfig

StateActionVec = TypeVar('StateActionVec')
State = TypeVar('State')
Action = TypeVar('Action')
Env = TypeVar('Env')


@dataclass(init=True, repr=True)
class EpsilonGreedyQEstimatorConfig(EpsilonGreedyConfig):
    gamma: float = 1.0
    alpha: float = 1.0
    env: Env = None


class EpsilonGreedyQEstimator(WithEstimatorMixin):
    """Q-function estimator using an epsilon-greedy policy
    for action selection
    """

    def __init__(self, config: EpsilonGreedyQEstimatorConfig):
        """Constructor

        Parameters
        ----------
        config: The instance configuration

        """
        super(EpsilonGreedyQEstimator, self).__init__()
        self.eps_policy: EpsilonGreedyPolicy = EpsilonGreedyPolicy.from_config(config)
        self.alpha: float = config.alpha
        self.gamma: float = config.gamma
        self.env: Env = config.env
        self.weights: np.array = None

    def q_hat_value(self, state_action_vec: StateActionVec) -> float:
        """Returns the
        :math: \hat{q}

        approximate value for the given state-action vector
        Parameters
        ----------
        state_action_vec

        Returns
        -------
        float


        """
        return self.weights.dot(state_action_vec)

    def update_weights(self, total_reward: float, state_action: Action,
                       state_action_: Action, t: float) -> None:
        """
        Update the weights
        Parameters
        ----------
        total_reward: The reward observed
        state_action: The action that led to the reward
        state_action_:
        t: The decay factor for alpha

        Returns
        -------

        None

        """
        v1 = self.q_hat_value(state_action_vec=state_action)
        v2 = self.q_hat_value(state_action_vec=state_action_)
        self.weights += self.alpha / t * (total_reward + self.gamma * v2 - v1) * state_action

    def on_state(self, state: State) -> Action:
        """Returns the action on the given state
        Parameters
        ----------
        state

        Returns
        -------

        """

        # compute the state values related to
        # the given state
        q_values = []

        for action in range(self.env.n_actions):
            state_action_vector = self.env.get_state_action_tile(action=action, state=state)
            q_values.append(state_action_vector)

        # choose an action at the current state
        action = self.eps_policy(q_values, state)
        return action
