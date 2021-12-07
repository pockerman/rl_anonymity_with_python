"""
Trainer
"""

from src.utils import INFO
from typing import TypeVar

Env = TypeVar("Env")
Agent = TypeVar("Agent")


class Trainer(object):

    def __init__(self, env: Env,  agent: Agent, configuration: dir) -> None:
        self.env = env
        self.agent = agent
        self.configuration = configuration

    def train(self):

        print("{0} Training agent {1}".format(INFO, self.agent.name))

        for episode in range(1, self.configuration["max_n_episodes"] + 1):
            print("INFO: Episode {0}/{1}".format(episode, self.configuration["max_n_episodes"]))

            # reset the environment
            ignore = self.env.reset()

            # train for a number of iterations
            self.agent.train(self.env)

            # is it time to update the model?
            if self.configuration["update_frequency"] % episode == 0:
                self.agent.update()


        print("{0} Training finished for agent {1}".format(INFO, self.agent.name))