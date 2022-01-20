"""
Trainer
"""

import numpy as np
from typing import TypeVar
from src.utils import INFO

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

    def actions_before_training(self):
        """
        Any actions to perform before training begins
        :return:
        """
        self.total_rewards: np.array = np.zeros(self.configuration['n_episodes'])
        self.iterations_per_episode = []

        self.agent.actions_before_training(self.env)

    def actions_after_episode_ends(self, **options):
        self.agent.actions_after_episode_ends(**options)

        if options["episode_idx"] % self.configuration['output_msg_frequency'] == 0:
            if self.env.config.distorted_set_path is not None:
                self.env.save_current_dataset(options["episode_idx"])

    def train(self):

        print("{0} Training agent {1}".format(INFO, self.agent.name))
        self.actions_before_training()

        for episode in range(0, self.configuration["n_episodes"]):
            print("{0} On episode {1}/{2}".format(INFO, episode, self.configuration["n_episodes"]))

            # reset the environment
            ignore = self.env.reset()

            # train for a number of iterations
            episode_score, total_distortion, n_itrs = self.agent.train(self.env)

            print("{0} Episode score={1}, episode total distortion {2}".format(INFO, episode_score, total_distortion / n_itrs))

            #if episode % self.configuration['output_msg_frequency'] == 0:
            print("{0} Episode finished after {1} iterations".format(INFO, n_itrs))

            self.iterations_per_episode.append(n_itrs)
            self.total_rewards[episode] = episode_score
            self.total_distortions.append(total_distortion)
            self.actions_after_episode_ends(**{"episode_idx": episode})

        print("{0} Training finished for agent {1}".format(INFO, self.agent.name))