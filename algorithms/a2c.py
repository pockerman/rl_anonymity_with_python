import numpy as np
from typing import TypeVar, Generic
import torch
import torch.nn as nn
import torch.nn.functional as F

Env = TypeVar("Env")
Optimizer = TypeVar("Optimizer")
State = TypeVar("State")
Action = TypeVar("Action")


class A2CNetBase(nn.Module):
    """
    Base class for A2C networks
    """

    def __init__(self, architecture):
        super(A2CNetBase, self).__init__()
        self.architecture = architecture

    def forward(self, x):
        return self.architecture(x)


class A2CNet(object):

    def __init__(self, common_net: A2CNetBase, policy_net: A2CNetBase, value_net: A2CNetBase):
        self.common_net, = common_net


class A2C(Generic[Optimizer]):

    def __init__(self, gamma: float, tau: float, n_workers: int,
                 n_iterations: int, optimizer: Optimizer):
        self.gamma = gamma
        self.tau = tau
        self.rewards = []
        self.n_workers = n_workers
        self.n_iterations = n_iterations
        self.optimizer = optimizer

    def _optimize_model(self):
        pass

    def select_action(self, state: State) -> Action:
        pass

    def train(self, env: Env) -> None:
        for episode in range(1, self.n_iterations + 1):
            pass
