"""
Trainer
"""

from typing import TypeVar

Env = TypeVar("Env")
Agent = TypeVar("Agent")


class Trainer(object):

    def __init__(self, env: Env,  agent: Agent, configuration: dir) -> None:
        self.env = env
        self.agent = agent
        self.configuration = configuration

    def train(self):

        for episode in range(1, self.configuration["max_n_episodes"] + 1):
            print("INFO: Episode {0}/{1}".format(episode, self.configuration["max_n_episodes"]))

            # reset the environment
            ignore = self.env.reset()

            # train for a number of iterations
            self.agent.train(self.env)
