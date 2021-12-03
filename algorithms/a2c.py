import numpy as np
from typing import TypeVar, Generic
import torch
import torch.nn as nn
import torch.nn.functional as F

Env = TypeVar("Env")
Optimizer = TypeVar("Optimizer")
State = TypeVar("State")
Action = TypeVar("Action")
TimeStep = TypeVar("TimeStep")


class A2CNetBase(nn.Module):
    """
    Base class for A2C networks
    """

    def __init__(self, architecture):
        super(A2CNetBase, self).__init__()
        self.architecture = architecture

    def forward(self, x):
        return self.architecture(x)


class A2CNet(nn.Module):

    def __init__(self, common_net: A2CNetBase, policy_net: A2CNetBase, value_net: A2CNetBase):
        super(A2CNet, self).__init__()
        self.common_net = common_net
        self.policy_net = policy_net
        self.value_net = value_net

    def forward(self, x):
        x = self.common_net(x)
        return self.policy_net(x), self.value_net(x)


class A2C(Generic[Optimizer]):

    def __init__(self, gamma: float, tau: float, n_workers: int,
                 n_iterations: int,  optimizer: Optimizer,
                 a2c_net: A2CNet):

        self.gamma = gamma
        self.tau = tau
        self.rewards = []
        self.n_workers = n_workers
        self.n_iterations = n_iterations
        self.optimizer = optimizer
        self.a2c_net = a2c_net
        self.name = "A2C"

    def _optimize_model(self):
        pass

    def select_action(self, observation: State) -> Action:
        pass

    def update(self):
        pass

    def train(self, env: Env) -> None:

        # reset the environment and obtain the
        # the time step
        time_step: TimeStep = env.reset()

        observation = time_step.observation

        for iteration in range(1, self.n_iterations + 1):

            # select an action
            action = self.select_action(observation=observation)

            # step in the environment according
            # to the selected action
            next_time_step = env.step(action=action)

            # we reached the end of the episode
            if next_time_step.last():
                break

            next_state = next_time_step.observation
            policy_val, v_val = self.a2c_net.forward(x=next_state)
            self._optimize_model()

