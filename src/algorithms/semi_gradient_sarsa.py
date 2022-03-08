"""Module semi_gradient_sarsa. Implements
episodic semi-gradient SARSA for estimating the state-action
value function. the im[plementation follows the algorithm
 at page 244 in the book by Sutton and Barto: Reinforcement Learning An Introduction
second edition 2020

"""

from dataclasses import  dataclass
from typing import TypeVar

from src.utils.mixins import WithMaxActionMixin, WithQTableMixinBase, WithEstimatorMixin
from src.utils.episode_info import EpisodeInfo
from src.spaces.time_step import TimeStep
from src.exceptions.exceptions import InvalidParamValue


Policy = TypeVar('Policy')
Env = TypeVar('Env')
State = TypeVar('State')
Action = TypeVar('Action')


@dataclass(init=True, repr=True)
class SemiGradSARSAConfig(object):
    """Configuration class for semi-gradient SARSA algorithm
    """
    gamma: float = 1.0
    alpha: float = 0.1
    n_itrs_per_episode: int = 100
    policy: Policy = None


class SemiGradSARSA(object):
    """SemiGradSARSA class. Implements the semi-gradient SARSA algorithm
    as described

    """

    def __init__(self, config: SemiGradSARSAConfig) -> None:
        self.config: SemiGradSARSAConfig = config

    @property
    def name(self) -> str:
        return "Semi-Grad SARSA"

    def actions_before_training(self, env: Env, **options) -> None:
        """Specify any actions necessary before training begins

        Parameters
        ----------

        env: The environment to train on
        options: Any key-value options passed by the client

        Returns
        -------

        None
        """

        self._validate()
        self._init()
        """
        for state in range(1, env.n_states):
            for action in range(env.n_actions):
                self.q_table[state, action] = 0.0
        """

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options) -> None:
        """Any actions to perform before the episode begins

        Parameters
        ----------

        env: The instance of the training environment
        episode_idx: The training episode index
        options: Any keyword options passed by the client code

        Returns
        -------

        None

        """

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Any actions after the training episode ends

        Parameters
        ----------

        env: The training environment
        episode_idx: The training episode index
        options: Any options passed by the client code

        Returns
        -------

        None
        """

    def on_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------

        env: The environment to train on
        options: Any keyword based options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo
        """

        episode_reward = 0.0
        episode_n_itrs = 0

        # reset the environment
        time_step = env.reset(**{"tiled_state": False})

        # select a state
        state: State = time_step.observation

        #choose an action using the policy
        action: Action = self.config.policy.on_state(state)

        for itr in range(self.config.n_itrs_per_episode):

            # take action and observe reward and next_state
            time_step: TimeStep = env.step(action, **{"tiled_state": False})

            reward: float = time_step.reward
            episode_reward += reward
            next_state: State = time_step.observation

            # if next_state is terminal i.e. the done flag
            # is set. then update the weights

            # otherwise chose next action as a function of q_hat
            next_action: Action = None
            # update the weights

            # update state
            state = next_state

            # update action
            action = next_action

            episode_n_itrs += 1

        episode_info = EpisodeInfo()
        episode_info.episode_score = episode_reward
        episode_info.episode_itrs = episode_n_itrs
        return episode_info

    def _weights_update_episode_done(self, state: State, reward: float,
                                     action: Action, next_state: State) -> None:
        """Update the weights due to the fact that
        the episode is finished

        Parameters
        ----------

        state: The current state
        reward: The reward to use
        action: The action we took at state
        next_state: The observed state

        Returns
        -------

        None
        """
        pass

    def _init(self) -> None:
        """
        Any initializations needed before starting the training

        Returns
        -------
        None
        """
        pass

    def _validate(self) -> None:
        """
        Validate the state of the agent. Is called before
        any training begins to check that the starting state is sane

        Returns
        -------

        None
        """

        if self.config is None:
            raise InvalidParamValue(param_name="self.config", param_value="None")

        if self.config.n_itrs_per_episode <= 0:
            raise ValueError("n_itrs_per_episode should be greater than zero")

        if not isinstance(self.config.policy, WithEstimatorMixin):
            raise InvalidParamValue(param_name="policy", param_value=str(self.config.policy))

