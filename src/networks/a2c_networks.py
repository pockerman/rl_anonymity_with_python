"""Module a2c_networks. Specifies various networks
for A2C algorithm

"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class A2CNetSimpleLinear(nn.Module):
    """A2CNetSimpleLinear. Specifies a network architecture
    consisting of three linear layers

    """

    def __init__(self, n_columns: int, n_actions: int):
        """Constructor.

        Parameters
        ----------
        n_columns: Number of columns
        n_actions: Number of actions

        """
        super(A2CNetSimpleLinear, self).__init__()
        self.linear_l1 = nn.Linear(in_features=n_columns, out_features=n_actions)
        self.actor = nn.Linear(in_features=n_actions, out_features=n_actions)
        self.critic = nn.Linear(n_actions, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Pass the state from the network

        Parameters
        ----------
        x: The torch tensor that represents the state

        Returns
        -------
        The actor and the critic values
        """

        # activate
        y = F.relu(self.linear_l1(x))
        actor = F.log_softmax(self.actor(y), dim=0)  # C
        critic = torch.tanh(self.critic(y))  # D

        return actor,  critic
