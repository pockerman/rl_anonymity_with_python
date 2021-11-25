import torch
import torch.nn as nn

class ActorNet(nn.Module):
    pass

class CriticNet(nn.Module):
    pass


class A2C(object):

    def __init__(self, env, n_episode_iterations: int,
                 update_frequency: int):
        self.env = env
        self.n_episode_iterations = n_episode_iterations
        self.update_frequency = update_frequency

    def select_action(self):
        """
        Select an action to execute
        :return: The action selected
        """
        pass

    def episode_step(self):

        for itr in range(1, self.n_episode_iterations + 1):

            # choose an action
            action = self.select_action()

            # step in the environment with that action
            # and receive new state and reward
            time_step = self.env.step(action)

            # formulate y

            # formulate gradients

            # update state

            # do we have to update the target network?
