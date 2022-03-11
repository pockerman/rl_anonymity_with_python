"""Module epsilon_greedy_q_estimator. Implements
a q-estimator by assuming linear function approximation

"""
from typing import TypeVar
import numpy as np
from dataclasses import dataclass

from src.utils.mixins import WithEstimatorMixin
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonGreedyConfig
from src.exceptions.exceptions import InvalidParamValue

StateActionVec = TypeVar('StateActionVec')
State = TypeVar('State')
Action = TypeVar('Action')
Env = TypeVar('Env')


@dataclass(init=True, repr=True)
class EpsilonGreedyQEstimatorConfig(EpsilonGreedyConfig):
    """Configuration class for EpsilonGreedyQEstimator

    """
    gamma: float = 1.0
    alpha: float = 1.0
    env: Env = None


class EpsilonGreedyQEstimator(WithEstimatorMixin):
    """Q-function estimator using an epsilon-greedy policy
    for action selection
    """

    def __init__(self, config: EpsilonGreedyQEstimatorConfig):
        """Constructor. Initialize the estimator with a given configuration

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
        self.initialize()

    def initialize(self) -> None:
        """Initialize the underlying weights

        Returns
        -------

        None

        """
        self.weights: np.array = np.zeros((self.env.n_states * self.env.n_actions))

    def q_hat_value(self, state_action_vec: StateActionVec) -> float:
        """Returns the
        :math: \hat{q}

        approximate value for the given state-action vector

        Parameters
        ----------

        state_action_vec: The state-action tiled vector

        Returns
        -------
        float

        """

        if self.weights is None:
            raise InvalidParamValue(param_name="weights", param_value="None. Have you called initialize?")

        return self.weights.dot(state_action_vec)

    def on_state(self, state: State) -> Action:
        """Returns the action on the given state

        Parameters
        ----------

        state: The state observed

        Returns
        -------

        An environment specific Action type
        """

        # get the approximation of the  q-values
        # given the state

        q_values = []

        for a in range(self.env.n_actions):
            tiled_vector = self.env.featurize_state_action(action=a, state=state)
            q_values.append(self.q_hat_value(tiled_vector))

        # choose an action at the current state
        action = self.eps_policy(q_values, state)

        # this is an integer get the ActionBase instead
        action = self.env.get_action(action)
        return action
