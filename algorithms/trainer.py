"""
Trainer
"""

class Trainer(object):

    def __init__(self, configuration) -> None:
        self.configuration = configuration

    def train(self):

        for episode in range(1, self.configuration["man_n_episodes"] + 1):
            print("INFO: Episode {0}/{1}".format(episode, self.configuration["man_n_episodes"]))

            # train for a number of iterations