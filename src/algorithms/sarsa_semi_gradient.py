"""
Implementation of SARSA semi-gradient algorithm.
Initial implementation is inspired from
https://michaeloneill.github.io/RL-tutorial.html
"""
import numpy as np
from typing import TypeVar

from src.utils.mixins import WithMaxActionMixin, WithQTableMixinBase
from src.utils.episode_info import EpisodeInfo
from src.algorithms.q_estimator import QEstimator
from src.exceptions.exceptions import InvalidParamValue

Env = TypeVar('Env')
Criterion = TypeVar('Criterion')
Policy = TypeVar('Policy')
Estimator = TypeVar('Estimator')


class SARSAnConfig:

    def __init__(self) -> None:
        self.gamma: float = 1.0
        self.alpha = 0.1
        self.n = 10
        self.n_itrs_per_episode: int = 100
        self.max_size: int = 4096
        self.use_trace: bool = False
        self.policy: Policy = None
        self.estimator: Estimator = None
        self.reset_estimator_z_traces: bool = False


class SARSAn(WithMaxActionMixin):
    """
    Implementation ofn-step  semi-gradient SARSA algorithm
    """

    def __init__(self, sarsa_config: SARSAnConfig):

        self.name = "SARSAn"
        self.config = sarsa_config
        self.q_table = {}

    def play(self, env: Env, stop_criterion: Criterion) -> None:
        pass

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
                reward = next_time_step.reward

                total_distortion += next_time_step.info["total_distortion"]
                episode_score += reward
                rewards.append(reward)

                if next_time_step.done:
                    T = itr + 1
                else:

                    next_action_idx = self.config.policy(self.q_table, next_state)
                    next_action = env.get_action(next_action_idx)
                    actions.append(next_action)

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

                # Update step
                self.config.estimator.update(states[update_time], actions[update_time], target)

            if update_time == T - 1:
                break

            counter += 1
            state = next_state
            action = next_action

        episode_info = EpisodeInfo()
        episode_info.episode_score = episode_score
        episode_info.total_distortion = total_distortion
        episode_info.info["m_iterations"] = counter
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

