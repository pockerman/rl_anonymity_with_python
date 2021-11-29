import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNetBase(nn.Module):
    """
    Base class for A2C networks
    """
    def __init__(self, architecture):
        super(A2CNetBase, self).__init__()
        self.architecture = architecture

    def forward(self, x):
        return self.architecture(x)




class A2C(object):

    def __init__(self, gamma: float, n_workers: int, tau:float,
                 max_n_steps: int, max_n_episodes: int,
                 common_net: A2CNetBase, policy_net: A2CNetBase, value_net: A2CNetBase,
                 optimizer):
        self.gamma = gamma
        self.rewards = []
        self.n_workers = n_workers
        self.optimizer = optimizer
        self.max_n_episodes = max_n_episodes

    def optimize_model(self):
        pass


    def train(self) -> None:


        for episode in range(self.max_n_episodes):

            # for each episode





