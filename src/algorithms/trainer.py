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
        self.total_rewards: np.array = None
        self.iterations_per_episode = []

    def actions_before_training(self):
        self.total_rewards: np.array = np.zeros(self.configuration['n_episodes'])
        self.iterations_per_episode = []

        self.agent.actions_before_training(self.env)

    def actions_after_episode_ends(self, **options):
        self.agent.actions_after_episode_ends(**options)

    def train(self):

        print("{0} Training agent {1}".format(INFO, self.agent.name))
        self.actions_before_training()

        for episode in range(0, self.configuration["n_episodes"]):
            print("INFO: Episode {0}/{1}".format(episode, self.configuration["n_episodes"]))

            # reset the environment
            ignore = self.env.reset()

            # train for a number of iterations
            episode_score, n_itrs = self.agent.train(self.env)

            if episode % self.configuration['output_msg_frequency'] == 0:
                print("{0}: On episode {1} training finished with  "
                      "{2} iterations. Total reward={3}".format(INFO, episode, n_itrs, episode_score))

            self.iterations_per_episode.append(n_itrs)
            self.total_rewards[episode] = episode_score

            # is it time to update the model?
            if self.configuration["update_frequency"] % episode == 0:
                self.agent.update()

            self.actions_after_episode_ends(**{"episode_idx": episode})

        print("{0} Training finished for agent {1}".format(INFO, self.agent.name))