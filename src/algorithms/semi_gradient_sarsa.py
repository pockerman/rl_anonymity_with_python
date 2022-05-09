"""Module semi_gradient_sarsa. Implements
episodic semi-gradient SARSA for estimating the state-action
value function. the im[plementation follows the algorithm
 at page 244 in the book by Sutton and Barto: Reinforcement Learning An Introduction
second edition 2020

"""

from dataclasses import dataclass
from typing import TypeVar

from src.utils.mixins import WithEstimatorMixin
from src.utils.episode_info import EpisodeInfo
from src.spaces.time_step import TimeStep
from src.utils.function_wraps import time_func_wrapper
from src.exceptions.exceptions import InvalidParamValue

Policy = TypeVar('Policy')
Env = TypeVar('Env')
State = TypeVar('State')
Action = TypeVar('Action')
Criterion = TypeVar('Criterion')


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

    def play(self, env: Env, stop_criterion: Criterion) -> None:
        """Play the agent on the environment. This should produce
        a distorted dataset

        Parameters
        ----------
        env: The environment to
        stop_criterion: The criteria to use to stop

        Returns
        -------
        None

        """
        # loop over the columns and for the
        # column get the action that corresponds to
        # the max payout.
        # TODO: This will no work as the distortion is calculated
        # by summing over the columns.

        total_dist = env.total_current_distortion()
        while stop_criterion.continue_itrs(total_dist):

            # use the policy to select an action
            state_idx = env.get_aggregated_state(total_dist)
            action_idx = self.config.policy.on_state(state_idx)
            action = env.get_action(action_idx)
            print("{0} At state={1} with distortion={2} select action={3}".format("INFO: ", state_idx, total_dist,
                                                                                  action.column_name + "-" + action.action_type.name))
            env.step(action=action)
            total_dist = env.total_current_distortion()

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
        episode_idx: The index of the training episode
        options: Any keyword based options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo
        """

        episode_info_, total_execution_time = self._do_train(env=env, episode_idx=episode_idx, **options)

        episode_info = EpisodeInfo()
        episode_info.episode_score = episode_info_.episode_score
        episode_info.episode_itrs = episode_info_.episode_itrs
        episode_info.total_distortion = episode_info_.total_distortion
        episode_info.total_execution_time = total_execution_time
        return episode_info

    @time_func_wrapper(show_time=False)
    def _do_train(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The index of the training episode
        options: Any keyword based options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo
        """

        episode_reward: float = 0.0
        episode_n_itrs: int = 0
        total_episode_distortion: float = 0.0

        # reset the environment
        time_step = env.reset(**{"tiled_state": False})

        # obtain the initial state S
        state: State = time_step.observation

        # initial action A
        action: Action = self.config.policy.on_state(state)

        for itr in range(self.config.n_itrs_per_episode):

            # take action A
            time_step: TimeStep = env.step(action, **{"tiled_state": False})

            # ... observe reward R
            reward: float = time_step.reward
            episode_reward += reward
            total_episode_distortion += time_step.info["total_distortion"]

            # ... observe the S prime
            next_state: State = time_step.observation

            # if next_state is terminal i.e. the done flag
            # is set. then update the weights
            if time_step.done:
                self._weights_update_episode_done(env=env, state=state, action=action, reward=reward)
                episode_n_itrs += 1
                break

            # choose action A prime as a function  of q_hat(S prime, *,  w)
            next_action: Action = self.config.policy.on_state(next_state)

            # update the weights. This expects tiled vector states
            self._weights_update(env=env, state=state, action=action,
                                 next_state=next_state, next_action=next_action, reward=reward)

            # update state
            state: State = next_state

            # update action
            action: Action = next_action

            episode_n_itrs += 1

        episode_info = EpisodeInfo()
        episode_info.episode_score = episode_reward
        episode_info.episode_itrs = episode_n_itrs
        episode_info.total_distortion = total_episode_distortion
        return episode_info

    def _weights_update_episode_done(self, env: Env, state: State, action: Action,
                                     reward: float, t: float = 1.0) -> None:
        """Update the weights of the underlying Q-estimator

        Parameters
        ----------

        state: The current state it is assumed to be a raw state
        reward: The reward observed when taking the given action when at the given state
        action: The action we took at the state


        Returns
        -------

        None
        """
        action_id = action
        if not isinstance(action, int):
            action_id = action.idx

        # get a copy of the weights
        weights = self.config.policy.weights

        tiled_state = env.featurize_state_action(action=action_id, state=state)
        v1 = self.config.policy.q_hat_value(state_action_vec=tiled_state)

        weights += self.config.alpha / t * (reward - v1) * tiled_state
        self.config.policy.weights = weights

    def _weights_update(self, env: Env, state: State, action: Action, reward: float,
                        next_state: State, next_action: Action, t: float = 1.0) -> None:
        """Update the weights due to the fact that
        the episode is finished

        Parameters
        ----------

        env: The environment instance that the training takes place
        state: The current state
        action: The action we took at state
        reward: The reward observed when taking the given action when at the given state
        next_state: The observed new state
        next_action: The action to be executed in next_state

        Returns
        -------

        None
        """

        action_id_1 = action
        if not isinstance(action, int):
            action_id_1 = action.idx

        action_id_2 = next_action
        if not isinstance(action, int):
            action_id_2 = next_action.idx

        # get a copy of the weights
        weights = self.config.policy.weights

        tiled_state1 = env.featurize_state_action(action=action_id_1, state=state)
        tiled_state2 = env.featurize_state_action(action=action_id_2, state=next_state)

        v1 = self.config.policy.q_hat_value(state_action_vec=tiled_state1)
        v2 = self.config.policy.q_hat_value(state_action_vec=tiled_state2)
        weights += self.config.alpha / t * (reward + self.config.gamma * v2 - v1) * tiled_state1
        self.config.policy.weights = weights

    def _init(self) -> None:
        """Any initializations needed before starting the training

        Returns
        -------

        None

        """

        if self.config.policy.weights is None or \
                len(self.config.policy.weights) == 0:
            self.config.policy.initialize()

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
