"""Module trainer. Specifies a utility class
for training serial reinforcement learning algorithms

"""

import numpy as np
from typing import TypeVar

from src.utils import INFO
from src.utils.function_wraps import time_func, time_func_wrapper
from src.utils.episode_info import EpisodeInfo

Env = TypeVar("Env")
Agent = TypeVar("Agent")


class Trainer(object):

    def __init__(self, env: Env,  agent: Agent, configuration: dir) -> None:
        """Constructor. Initialize a trainer by passing the training environment
        instance the agen to train and configuration dictionary

        Parameters
        ----------

        env: The environment to train the agent
        agent: The agent to train
        configuration: Configuration parameters for the trainer

        """
        self.env = env
        self.agent = agent
        self.configuration = configuration
        # monitor performance
        self.total_rewards: np.array = np.zeros(configuration['n_episodes'])
        self.iterations_per_episode = []
        self.total_distortions = []

    def avg_rewards(self) -> np.array:
        """
        Returns the average reward per episode
        :return:
        """
        avg = np.zeros(self.configuration['n_episodes'])

        for i in range(self.total_rewards.shape[0]):
            avg[i] = self.total_rewards[i] / self.iterations_per_episode[i]
        return avg

    def avg_distortion(self) -> np.array:
        """
        Returns the average reward per episode
        :return:
        """
        avg = np.zeros(self.configuration['n_episodes'])

        for i in range(len(self.total_distortions)):
            avg[i] = self.total_distortions[i] / self.iterations_per_episode[i]
        return avg

    def actions_before_training(self) -> None:
        """Any actions to perform before training begins

        Returns
        -------

        None
        """

        self.total_rewards: np.array = np.zeros(self.configuration['n_episodes'])
        self.iterations_per_episode = []
        self.agent.actions_before_training(self.env)

    def actions_before_episode_begins(self, env: Env, episode_idx: int,  **options) -> None:
        """Perform any actions necessary before the training begins

        Parameters
        ----------
        env: The environment to train on
        episode_idx: The training episode index
        options: Any options passed by the client code

        Returns
        -------

        None

        """
        self.agent.actions_before_episode_begins(env, episode_idx, **options)

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Any actions after the training episode ends

        Parameters
        ----------

        env: The environment to train on
        episode_idx:  The training episode index
        options: Any options passed by the client code

        Returns
        -------

        None
        """
        self.agent.actions_after_episode_ends(env, episode_idx, **options)

        if episode_idx % self.configuration['output_msg_frequency'] == 0:
            if self.env.config.distorted_set_path is not None:
                self.env.save_current_dataset(episode_idx)

    @time_func_wrapper(show_time=True)
    def train(self):

        print("{0} Training agent {1}".format(INFO, self.agent.name))
        self.actions_before_training()

        for episode in range(0, self.configuration["n_episodes"]):
            print("{0} On episode {1}/{2}".format(INFO, episode, self.configuration["n_episodes"]))

            self.actions_before_episode_begins(self.env, episode)

            # train for a number of iterations
            episode_info: EpisodeInfo = self.agent.on_episode(self.env, episode)

            print("{0} Episode {1} finished in {2} secs".format(INFO, episode, episode_info.total_execution_time))
            print("{0} Episode score={1}, episode total avg distortion {2}".format(INFO, episode_info.episode_score,
                                                                               episode_info.total_distortion / episode_info.episode_itrs))

            print("{0} Episode finished after {1} iterations".format(INFO, episode_info.episode_itrs))

            self.iterations_per_episode.append(episode_info.episode_itrs)
            self.total_rewards[episode] = episode_info.episode_score
            self.total_distortions.append(episode_info.total_distortion)
            self.actions_after_episode_ends(self.env, episode, **{})

        print("{0} Training finished for agent {1}".format(INFO, self.agent.name))
