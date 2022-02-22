"""
Trainer
"""

import numpy as np
from typing import TypeVar

from src.utils import INFO
from src.utils.function_wraps import time_func
from src.utils.episode_info import EpisodeInfo

Env = TypeVar("Env")
Agent = TypeVar("Agent")


class Trainer(object):

    def __init__(self, env: Env,  agent: Agent, configuration: dir) -> None:
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

    def actions_before_training(self):
        """
        Any actions to perform before training begins
        :return:
        """
        self.total_rewards: np.array = np.zeros(self.configuration['n_episodes'])
        self.iterations_per_episode = []

        self.agent.actions_before_training(self.env)

    def actions_before_episode_begins(self, **options) -> None:
        """
        Perform any actions necessary before the training begins
        :param options:
        :return:
        """
        self.agent.actions_before_episode_begins(**options)

    def actions_after_episode_ends(self, **options):
        self.agent.actions_after_episode_ends(**options)

        if options["episode_idx"] % self.configuration['output_msg_frequency'] == 0:
            if self.env.config.distorted_set_path is not None:
                self.env.save_current_dataset(options["episode_idx"])

    @time_func
    def train(self):

        print("{0} Training agent {1}".format(INFO, self.agent.name))
        self.actions_before_training()

        for episode in range(0, self.configuration["n_episodes"]):
            print("{0} On episode {1}/{2}".format(INFO, episode, self.configuration["n_episodes"]))

            # reset the environment
            #ignore = self.env.reset()

            self.actions_before_episode_begins(**{"env": self.env})
            # train for a number of iterations
            #episode_score, total_distortion, n_itrs = self.agent.on_episode(self.env)
            episode_info: EpisodeInfo = self.agent.on_episode(self.env)

            print("{0} Episode score={1}, episode total avg distortion {2}".format(INFO, episode_info.episode_score,
                                                                               episode_info.total_distortion / episode_info.info["n_iterations"]))

            #if episode % self.configuration['output_msg_frequency'] == 0:
            print("{0} Episode finished after {1} iterations".format(INFO, episode_info.info["n_iterations"]))

            self.iterations_per_episode.append(episode_info.info["n_iterations"])
            self.total_rewards[episode] = episode_info.episode_score
            self.total_distortions.append(episode_info.total_distortion)
            self.actions_after_episode_ends(**{"episode_idx": episode})

        print("{0} Training finished for agent {1}".format(INFO, self.agent.name))
