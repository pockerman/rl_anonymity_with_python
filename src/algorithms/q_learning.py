"""
Simple Q-learning algorithm
"""

import numpy as np
from typing import TypeVar
from dataclasses import dataclass

from src.exceptions.exceptions import InvalidParamValue
from src.utils.mixins import WithMaxActionMixin, WithQTableMixinBase
from src.utils.episode_info import EpisodeInfo
from src.utils.function_wraps import time_func_wrapper
from src.spaces.env_type import DiscreteEnvType

Env = TypeVar('Env')
Policy = TypeVar('Policy')
Criterion = TypeVar('Criterion')


@dataclass(init=True, repr=True)
class QLearnConfig(object):
    """Configuration  for Q-learning"""

    gamma: float = 1.0
    alpha: float = 0.1
    n_itrs_per_episode: int = 100
    policy: Policy = None


class QLearning(WithMaxActionMixin):
    """Q-learning algorithm implementation

    """

    def __init__(self, algo_config: QLearnConfig):
        """Constructor. Construct an instance of the algorithm
        by passing the configuration parameters

        Parameters
        ----------
        algo_config: The configuration parameters

        """

        super(QLearning, self).__init__()
        self.q_table = {}
        self.config = algo_config

    @property
    def name(self) -> str:
        return "QLearning"

    def actions_before_training(self, env: Env, **options) -> None:
        """Any actions before training begins

        Parameters
        ----------
        env: The environment that training occurs
        options: Any options passed by the client code

        Returns
        -------
        None
        """

        if not isinstance(self.config.policy, WithQTableMixinBase):
            raise InvalidParamValue(param_name="policy", param_value=str(self.config.policy))

        if env.env_type == DiscreteEnvType.MULTI_COLUMN_STATE:

            if len(env.state_space) == 0:
                raise ValueError("The state space is empty")

            for state in env.state_space:
                for action in range(env.n_actions):
                    self.q_table[state, action] = 0.0
        else:

            for state in range(1, env.n_states + 1):
                for action in range(env.n_actions):
                    self.q_table[state, action] = 0.0

    def actions_before_episode_begins(self, env: Env, episode_idx, **options) -> None:
        """Execute any actions the algorithm needs before
        the episode ends

        Parameters
        ----------

        env: The environment that training occurs
        episode_idx: The episode index
        options: Any options passed by the client code

        Returns
        -------

        None
        """

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs after
        the episode ends

        Parameters
        ----------

        env: The environment that training occurs
        episode_idx: The episode index
        options: Any options passed by the client code

        Returns
        -------
        None
        """
        self.config.policy.actions_after_episode(episode_idx)

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

        # set the q_table for the policy
        self.config.policy.q_table = self.q_table
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

    def on_episode(self, env: Env, episode_idx: int,  **options) -> EpisodeInfo:
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

        episode_info, total_time = self._do_train(env, episode_idx, **options)
        episode_info.total_execution_time = total_time
        return episode_info

    @time_func_wrapper(show_time=False)
    def _do_train(self, env: Env, episode_idx: int, **option) -> EpisodeInfo:
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
            self._update_q_table(state=state, action=action_idx, reward=reward,
                                 next_state=next_state, n_actions=env.n_actions)
            state = next_state  # S <- S'
            counter += 1
            total_distortion += next_time_step.info["total_distortion"]

            if next_time_step.last():
                break

        episode_info = EpisodeInfo(episode_score=episode_score, total_distortion=total_distortion, episode_itrs=counter)
        return episode_info

    def _update_q_table(self, state: int, action: int, n_actions: int,
                        reward: float, next_state: int = None) -> None:
        """ Update the tabular state-action function

        Parameters
        ----------
        state: State observed
        action: The action taken
        n_actions: Number of actions in the data set
        reward: The reward observed
        next_state: The next state observed

        Returns
        -------
        None

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
